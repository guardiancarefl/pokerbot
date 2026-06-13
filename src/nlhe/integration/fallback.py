"""Guaranteed-action fallback watchdog (Stage 2, operator-approved
2026-06-11; design recorded in docs/SESSION_LOG.md + the session-1 audit).

Problem (first live session, log 20260611_163815): 8 hero-to-act spots
were never decided — scraper-suspect bursts and replay failures left
hero hanging 13.8-35.1s until the operator or the site timer resolved
them. The decision pipeline's failure mode is SILENCE, not wrong
output.

The watchdog guarantees an action plan exists within N seconds of
hero-to-act detection: CHECK if a CHECK button is available, FOLD
otherwise. A fired fallback is logged loudly as a FALLBACK — never as
a model decision. It never emits CALL or RAISE: CHECK/FOLD commit no
chips, which is the entire trust argument (same as the SAFE-FOLD
invariant's safe_action).

Strict isolation from the live decision path:
  - lives entirely in the listener loop (run_live_dryrun.py), never
    inside make_decision / live_loop.py;
  - never reads or writes DecisionCache (a fallback is not a decision
    identity) or SessionTracker (no anchor/pot state);
  - consumes only the raw scraper record + the decision STATUS string;
  - time is injected by the caller (`now`), so the class is pure and
    deterministic under test.

State machine (per to-act EPISODE):
  ARM    on the first frame where hero has a real action UI
         (CHECK/CALL/RAISE/BET buttons — the predicate validated by
         scripts/decision_audit.py on session 163815: matched all 15
         missed to-act frames, excluded all 23 FOLD-only muck-UI
         frames) and the pipeline produced no decision. Deadline =
         arrival time of THAT frame + fallback_seconds; later frames
         refresh the button/controls snapshot but never the deadline.
  DISARM when a decision* status arrives for the hand, when the real
         action UI is absent for `vanish_frames` consecutive frames
         (action was taken externally; a 1-frame scraper flicker does
         NOT reset the deadline), or when a new hand is dealt.
  FIRE   once per episode when now >= deadline, on a to-act frame or
         framelessly via poll() (heartbeat/tick path — a frozen
         scraper cannot suppress the guarantee). After firing the
         episode is dead until a disarm event allows re-arming (a
         later street in the same hand is a fresh episode).

Session-abort criterion (recommendation only; the runner displays it,
nothing here clicks or sits out): >= `abort_fallbacks` fallback hands
within any `abort_window_hands`-hand window, OR fallbacks in
`abort_consecutive` consecutive hands. Scraper degradation is bursty —
session 163815 lost 3 hands in a 4-hand stretch (seqs 346-370), which
trips the consecutive rule.

Abort bookkeeping counts EVIDENCE HANDS only (2026-06-13 fix, session
230149 postmortem [2]): a hand enters the abort counters (`_hand_flags`
/ `hands_seen`) only if at least one of its frames showed a real
hero-to-act UI (or a fallback fired in it). Hand boundaries are still
keyed on raw hero-card tuples — but a phantom "hand" (single-frame
card-flicker segmentation artifact, e.g. the 1-frame ('2c',) read at
seq 1381 between the TsAc fallback hand and the very next real
fallback hand) carries no hero-to-act evidence and is now IGNORED by
the counters instead of inserting a clean entry that resets the
consecutive rule (fallbacks #6+#7 should have tripped the enforced
abort; they did not). Mirrors the triage segmentation's pair-less-
fragment merge rule (scripts/dryrun_triage.py segment_hands): fragments
without the defining evidence don't stand as hands. Deliberate
corollary: a real hand in which hero never had a decision (e.g. a BB
walk — muck-only UI at most) no longer resets the consecutive counter
either; it carries no evidence the pipeline recovered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

REAL_ACTION_BUTTONS = {"CHECK", "CALL", "RAISE", "BET", "ALLIN"}


def _buttons(raw_record: dict) -> list[str]:
    action = raw_record.get("action") or {}
    return [str(b).upper() for b in (action.get("buttons") or [])]


def is_real_to_act(raw_record: dict) -> bool:
    """Hero has a genuine decision UI. FOLD-only button sets are the
    muck / fold-ahead-of-action UI and do NOT count."""
    btns = _buttons(raw_record)
    return any(b in REAL_ACTION_BUTTONS for b in btns)


@dataclass
class FallbackPlan:
    """A fired guaranteed-action fallback. NOT a model decision."""
    action_kind: str                    # "check" | "fold"
    hand_cards: tuple[str, ...]
    armed_seq: int | None
    armed_captured_at: str
    fired_after_seq: int | None         # None when fired by poll()
    waited_seconds: float
    check_available: bool
    last_controls: dict = field(repr=False, default_factory=dict)
    n_fallbacks_session: int = 0
    abort_recommended: bool = False
    abort_reason: str | None = None

    def as_log_record(self) -> dict:
        """JSONL row. Deliberately carries NO `seq` key and NO
        `raw_record` key so existing replay/triage tooling
        (dryrun_triage filters on "seq" in rec; decision_audit and
        replay_make_decision_diff filter on raw_record presence)
        ignores fallback rows by construction."""
        return {
            "record_type": "fallback",
            "action_kind": self.action_kind,
            "hand_cards": list(self.hand_cards),
            "armed_seq": self.armed_seq,
            "armed_captured_at": self.armed_captured_at,
            "fired_after_seq": self.fired_after_seq,
            "waited_seconds": round(self.waited_seconds, 3),
            "check_available": self.check_available,
            "n_fallbacks_session": self.n_fallbacks_session,
            "abort_recommended": self.abort_recommended,
            "abort_reason": self.abort_reason,
        }


class FallbackWatchdog:
    def __init__(
        self,
        fallback_seconds: float,
        vanish_frames: int = 2,
        abort_window_hands: int = 10,
        abort_fallbacks: int = 3,
        abort_consecutive: int = 2,
        hand_deadline: bool = False,
        click_confirmation_mode: bool = False,
    ) -> None:
        """v2 options (approved 2026-06-11, both OFF by default = v1):

        hand_deadline: the deadline anchors to the FIRST to-act
          evidence for the SPOT — keyed (hero_cards, board) — and
          survives controls flicker / vanish-disarm / re-arm cycles.
          Closes the 9cAd gap (session 204751 seqs 1013-1020:
          intermittent button visibility reset the per-episode
          deadline; the spot hung ~30s without firing). The spot
          anchor clears when a decision* is emitted (spot resolved),
          when the hand changes, or after a fire (a re-appearing
          already-fired spot gets a FRESH deadline, never an instant
          re-fire).

        click_confirmation_mode: Stage-2 executor mode — a decision*
          status NO LONGER disarms; only controls vanishing (the click
          actually landed / action taken) or a hand change does. A
          silently-failed click leaves the buttons up past the
          deadline and the watchdog fires the safe action. The fired
          action stays CHECK-if-free/FOLD regardless of what the
          decision was — the fallback never escalates.
        """
        if fallback_seconds <= 0:
            raise ValueError("fallback_seconds must be > 0; gate the "
                             "watchdog's construction on the flag instead")
        self.fallback_seconds = float(fallback_seconds)
        self.vanish_frames = int(vanish_frames)
        self.abort_window_hands = int(abort_window_hands)
        self.abort_fallbacks = int(abort_fallbacks)
        self.abort_consecutive = int(abort_consecutive)
        self.hand_deadline = bool(hand_deadline)
        self.click_confirmation_mode = bool(click_confirmation_mode)

        # episode state
        self._armed: dict | None = None
        self._fired_this_episode = False
        self._vanish_count = 0
        # v2 spot-level deadline anchor (hand_deadline mode)
        self._spot_key: tuple | None = None
        self._spot_t0: float | None = None
        # hand tracking
        self._current_hand: tuple[str, ...] | None = None
        self._current_hand_fallback = False
        # True iff the CURRENT hand has shown at least one real
        # hero-to-act frame (or fired a fallback). Hands without this
        # evidence are phantom/no-decision hands and never enter the
        # abort counters (see module docstring, 2026-06-13 fix).
        self._current_hand_evidence = False
        self._hand_flags: list[bool] = []   # closed EVIDENCE hands, append order
        # session counters
        self.n_fallbacks = 0
        self.hands_seen = 0                 # evidence hands only
        self.abort_recommended = False
        self.abort_reason: str | None = None

    # ── hand / episode bookkeeping ─────────────────────────────────────

    def _close_hand(self) -> None:
        # Phantom-hand guard: a hand with NO hero-to-act evidence (e.g.
        # a single-frame card-flicker tuple) is dropped from the abort
        # bookkeeping entirely — appending a clean False here is what
        # reset the consecutive counter between real fallback hands
        # (session 230149 seq 1381, fallbacks #6/#7).
        if self._current_hand is not None and self._current_hand_evidence:
            self._hand_flags.append(self._current_hand_fallback)
        self._current_hand_fallback = False
        self._current_hand_evidence = False

    def _mark_hand_evidence(self) -> None:
        if not self._current_hand_evidence:
            self._current_hand_evidence = True
            self.hands_seen += 1

    def _on_new_hand(self, cards: tuple[str, ...]) -> None:
        self._close_hand()
        self._current_hand = cards
        self._disarm()
        self._clear_spot()

    def _disarm(self) -> None:
        self._armed = None
        self._fired_this_episode = False
        self._vanish_count = 0

    def _clear_spot(self) -> None:
        self._spot_key = None
        self._spot_t0 = None

    def _check_abort(self) -> None:
        flags = self._hand_flags + [self._current_hand_fallback]
        window = flags[-self.abort_window_hands:]
        if sum(window) >= self.abort_fallbacks:
            self.abort_recommended = True
            self.abort_reason = (
                f"{sum(window)} fallback hands in the last "
                f"{len(window)} hands (threshold "
                f"{self.abort_fallbacks}/{self.abort_window_hands})")
            return
        tail = flags[-self.abort_consecutive:]
        if len(tail) == self.abort_consecutive and all(tail):
            self.abort_recommended = True
            self.abort_reason = (
                f"fallbacks in {self.abort_consecutive} consecutive hands")

    # ── main entry points ──────────────────────────────────────────────

    def observe(self, raw_record: dict, status: str,
                seq: int | None, now: float) -> Optional[FallbackPlan]:
        """Feed one processed frame. `status` is LiveDecision.status;
        `now` is the listener's wall-clock arrival time. Returns a
        FallbackPlan iff the deadline expired on this frame."""
        cards = tuple(raw_record.get("hero_cards") or [])
        if cards and cards != self._current_hand:
            self._on_new_hand(cards)

        to_act = is_real_to_act(raw_record)
        if to_act:
            # A real action UI is the hand's hero-to-act evidence —
            # marked before the decision-disarm early-return so decided
            # hands count too.
            self._mark_hand_evidence()

        if status.startswith("decision") and not self.click_confirmation_mode:
            # Pipeline produced/locked an action — the guarantee holds.
            # (click_confirmation_mode: a decision is not enough; only
            # the click landing — controls vanishing — disarms.)
            self._disarm()
            self._clear_spot()
            return None

        if not to_act:
            if self._armed is not None or self._fired_this_episode:
                self._vanish_count += 1
                if self._vanish_count >= self.vanish_frames:
                    # Action was taken externally / street ended.
                    # hand_deadline mode: the SPOT anchor survives a
                    # vanish-disarm — if the same (cards, board) spot
                    # re-appears, it resumes the original deadline.
                    self._disarm()
            return None

        # Real to-act frame with no (confirmed) action.
        self._vanish_count = 0
        if self._fired_this_episode:
            return None
        btns = _buttons(raw_record)
        check_available = "CHECK" in btns
        controls = raw_record.get("controls") or {}
        if self._armed is None:
            t0 = float(now)
            if self.hand_deadline:
                spot_key = (tuple(cards), tuple(raw_record.get("board") or ()))
                if spot_key == self._spot_key and self._spot_t0 is not None:
                    t0 = self._spot_t0      # resume the spot's deadline
                else:
                    self._spot_key = spot_key
                    self._spot_t0 = t0
            self._armed = {
                "t0": t0,
                "seq": seq,
                "captured_at": raw_record.get("captured_at", ""),
                "check_available": check_available,
                "controls": controls,
            }
        else:
            if self.hand_deadline:
                spot_key = (tuple(cards), tuple(raw_record.get("board") or ()))
                if spot_key != self._spot_key:
                    # Street advanced while armed (no vanish in between):
                    # a NEW spot — fresh deadline, new anchor.
                    self._spot_key = spot_key
                    self._spot_t0 = float(now)
                    self._armed["t0"] = float(now)
                    self._armed["seq"] = seq
                    self._armed["captured_at"] = raw_record.get(
                        "captured_at", "")
            # Refresh UI snapshot; never the deadline (within a spot).
            self._armed["check_available"] = check_available
            self._armed["controls"] = controls
        if now - self._armed["t0"] >= self.fallback_seconds:
            return self._fire(fired_after_seq=seq, now=now)
        return None

    def poll(self, now: float) -> Optional[FallbackPlan]:
        """Frameless deadline check (heartbeat / socket-tick path).
        Fires with the last-seen controls snapshot even if the scraper
        has gone quiet — a frozen scraper must not suppress the
        guarantee."""
        if self._armed is None or self._fired_this_episode:
            return None
        if now - self._armed["t0"] < self.fallback_seconds:
            return None
        return self._fire(fired_after_seq=None, now=now)

    def _fire(self, fired_after_seq: int | None, now: float) -> FallbackPlan:
        armed = self._armed or {}
        self._fired_this_episode = True
        # A fired spot loses its anchor: if the same spot re-appears
        # after a disarm, it gets a FRESH deadline (no instant re-fire).
        self._clear_spot()
        self.n_fallbacks += 1
        self._current_hand_fallback = True
        # A fired fallback is hero-to-act evidence by construction (the
        # episode armed on a real action UI); poll()-fired plans may
        # never have routed a to-act frame through observe() for the
        # current card-tuple, so latch it here too.
        self._mark_hand_evidence()
        self._check_abort()
        return FallbackPlan(
            action_kind="check" if armed.get("check_available") else "fold",
            hand_cards=self._current_hand or (),
            armed_seq=armed.get("seq"),
            armed_captured_at=armed.get("captured_at", ""),
            fired_after_seq=fired_after_seq,
            waited_seconds=float(now) - float(armed.get("t0", now)),
            check_available=bool(armed.get("check_available")),
            last_controls=armed.get("controls") or {},
            n_fallbacks_session=self.n_fallbacks,
            abort_recommended=self.abort_recommended,
            abort_reason=self.abort_reason,
        )


class AbortGate:
    """Enforced session-abort state (Stage-2 wiring, approved 2026-06-11).

    The FallbackWatchdog only RECOMMENDS abort. When the listener runs
    with --abort-enforce, this gate turns the recommendation into an
    enforced state: while active, the listener suppresses click plans
    (decisions are still computed, displayed and logged — the EXECUTION
    surface is what's gated) and banners SIT OUT NOW on every decision.

    Reset is manual and out-of-band: the operator creates the reset
    file (`touch <reset_path>`); the gate clears on the next frame and
    deletes the file. No auto-sit-out click in this stage (Stage-2b).
    """

    def __init__(self, reset_path: str) -> None:
        self.reset_path = str(reset_path)
        self.active = False
        self.reason: str | None = None
        self.tripped_count = 0

    def trip(self, reason: str | None) -> bool:
        """Activate. Returns True iff this call newly activated the gate."""
        if self.active:
            return False
        self.active = True
        self.reason = reason or "abort criterion tripped"
        self.tripped_count += 1
        return True

    def check_reset(self) -> bool:
        """Clear the gate if the operator's reset file exists. Returns
        True iff the gate was cleared by this call."""
        if not self.active:
            return False
        import os
        if os.path.exists(self.reset_path):
            try:
                os.unlink(self.reset_path)
            except OSError:
                pass
            self.active = False
            self.reason = None
            return True
        return False
