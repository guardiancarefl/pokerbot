"""Click target computation for the staged auto-clicker.

Given a `client_action` dict (from the OpenSpiel-chip-int → real-table
translation already used by `run_logonly_resolver.py`) and the scraper's
`controls` block, produces a `ClickPlan` describing the exact click /
keystroke sequence the auto-clicker would execute to perform the
action on the real table.

Stage 1 (DRY-RUN): returns the plan; the harness DISPLAYS + LOGS it
but never executes. Boxes are normalized [0..1] floats per the
scraper's existing convention. Stage 2 introduces a separate denormalize
step that takes the table window's pixel origin + size and converts
boxes to absolute pixel coords.

Action → plan mapping (Stage 1):
  fold     → 1 click  on the FOLD button box
  check    → 1 click  on the CHECK button box
  call     → 1 click  on the CALL button box
  raise_to → 3 steps: click bet_input → type chip_amount → click RAISE

Per the Stage-1 decision (uniform code path; see protocol discussion),
raises always go through the bet_input field. Shortcut buttons
(X3/X5/POT/ALL-IN) are NOT used; the chip amount is typed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


Box = tuple[float, float, float, float]  # x1, y1, x2, y2 normalized 0..1


@dataclass
class ClickStep:
    """One click or keystroke step in a click plan."""
    kind: Literal["click", "type"]
    # For "click": UI element label (e.g. "FOLD", "RAISE", "bet_input").
    # For "type": same label string identifying the input target.
    target: str
    # Normalized [x1,y1,x2,y2] box. Required for "click"; may be None for
    # "type" (target is implied by the prior click).
    box: Box | None = None
    # For "type" steps: the literal text to type (chip amount as string).
    payload: str | None = None


@dataclass
class ClickPlan:
    """Sequence of click/keystroke steps the auto-clicker would execute.

    Stage 1: produced and DISPLAYED only. Steps with `box=None` are valid
    only for "type" entries; downstream stages execute them in order.

    When `reason` is set, `steps` is empty — the plan is a no-op
    (safe-fold or missing UI element). The reason is shown in the
    dashboard so the operator can spot read issues live.
    """
    action_kind: str  # "fold" | "check" | "call" | "raise_to" | "no_plan"
    chip_amount: int | None
    steps: list[ClickStep] = field(default_factory=list)
    reason: str | None = None

    def is_no_op(self) -> bool:
        return not self.steps


def _box_for_button(label: str, action_buttons) -> Box | None:
    """Find the box for the named button (case-insensitive)."""
    if not action_buttons:
        return None
    target = label.upper()
    for b in action_buttons:
        if (b.get("label") or "").upper() == target:
            box = b.get("box")
            if box is not None and len(box) == 4:
                return (float(box[0]), float(box[1]),
                        float(box[2]), float(box[3]))
    return None


def compute_click_target(
    client_action: dict,
    controls: dict | None,
) -> ClickPlan:
    """Compute the click plan for the bot's intended action.

    Args:
        client_action: dict with keys "kind" ("fold"/"check"/"call"/
            "raise_to"), "chip_amount" (int or None for non-raises),
            "raw_openspiel_chip_int" (int — audit field, not used here).
        controls: scraper record's `controls` dict. Must contain
            `action_buttons` (list of {label, box, amount?}) and, for
            raises, `bet_input` (dict with `box`).

    Returns:
        ClickPlan. For supported actions: steps populated. For
        unsupported / missing UI elements: empty steps + a `reason`
        explaining why no plan could be formed (so the dashboard can
        flag a probable read issue).
    """
    if controls is None:
        controls = {}
    kind = client_action.get("kind")
    chip_amount = client_action.get("chip_amount")
    action_buttons = controls.get("action_buttons") or []

    if kind == "fold":
        box = _box_for_button("FOLD", action_buttons)
        if box is None:
            return ClickPlan("fold", None, [],
                              reason="FOLD button not found in scraper controls")
        return ClickPlan("fold", None,
                          [ClickStep("click", "FOLD", box=box)])

    if kind == "check":
        box = _box_for_button("CHECK", action_buttons)
        if box is None:
            return ClickPlan("check", None, [],
                              reason="CHECK button not found in scraper controls")
        return ClickPlan("check", None,
                          [ClickStep("click", "CHECK", box=box)])

    if kind == "call":
        box = _box_for_button("CALL", action_buttons)
        if box is None:
            return ClickPlan("call", chip_amount, [],
                              reason="CALL button not found in scraper controls")
        return ClickPlan("call", chip_amount,
                          [ClickStep("click", "CALL", box=box)])

    if kind == "raise_to":
        if chip_amount is None:
            return ClickPlan("raise_to", None, [],
                              reason="raise_to but chip_amount is None")
        bet_input_box_raw = (controls.get("bet_input") or {}).get("box")
        # The action button is labeled RAISE on streets where there's
        # already a bet (preflop facing BB, postflop facing a bet) and
        # BET on a postflop street with no prior bet (hero opens). Both
        # map to our internal "raise_to" action — accept either.
        raise_box = _box_for_button("RAISE", action_buttons)
        raise_label = "RAISE"
        if raise_box is None:
            raise_box = _box_for_button("BET", action_buttons)
            raise_label = "BET"
        if bet_input_box_raw is None or len(bet_input_box_raw) != 4:
            return ClickPlan("raise_to", chip_amount, [],
                              reason="bet_input box missing from scraper controls")
        if raise_box is None:
            return ClickPlan("raise_to", chip_amount, [],
                              reason="neither RAISE nor BET button found in scraper controls")
        bet_input_box: Box = (
            float(bet_input_box_raw[0]), float(bet_input_box_raw[1]),
            float(bet_input_box_raw[2]), float(bet_input_box_raw[3]),
        )
        return ClickPlan("raise_to", chip_amount, [
            ClickStep("click", "bet_input", box=bet_input_box),
            ClickStep("type", "bet_input", payload=str(int(chip_amount))),
            ClickStep("click", raise_label, box=raise_box),
        ])

    return ClickPlan(kind or "unknown", chip_amount, [],
                      reason=f"unsupported action kind: {kind!r}")


def click_plan_for_safe_fold(reason: str) -> ClickPlan:
    """The no-op plan returned when a frame is safe-folded (invariant
    fail, replay error, missing tracker pre-hand, etc.). The Stage 1
    dashboard shows this with the SAFE-FOLD banner."""
    return ClickPlan("no_plan", None, [], reason=reason)


def plan_as_dict(plan: ClickPlan) -> dict:
    """Serializable form (for JSONL logging)."""
    return {
        "action_kind": plan.action_kind,
        "chip_amount": plan.chip_amount,
        "reason": plan.reason,
        "steps": [
            {
                "kind": s.kind,
                "target": s.target,
                "box": list(s.box) if s.box else None,
                "payload": s.payload,
            }
            for s in plan.steps
        ],
    }
