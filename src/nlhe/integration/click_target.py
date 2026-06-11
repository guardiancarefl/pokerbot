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
    kind: Literal["click", "type", "verify"]
    # For "click": UI element label (e.g. "FOLD", "RAISE", "bet_input").
    # For "type"/"verify": same label string identifying the input target.
    target: str
    # Normalized [x1,y1,x2,y2] box. Required for "click"; may be None for
    # "type" (target is implied by the prior click).
    box: Box | None = None
    # For "type" steps: the literal text to type (chip amount as string).
    # For "verify" steps: the text the executor must READ BACK from the
    # input box before proceeding — mismatch aborts the plan (extended
    # mode only; Stage-2 typed-amount verification).
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
    # Extended mode only: set when the intended action could not be
    # executed literally and was realized as the closest legal UI action
    # (e.g. raise_to at a call-only facing-all-in UI -> "call";
    # all-in raise via the ALLIN button -> "allin_button";
    # call with no CALL button but CHECK present -> "check").
    # None in default mode — plan_as_dict omits the key when None so
    # default-mode serialization is byte-identical to pre-extended code.
    realized_as: str | None = None

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
    extended: bool = False,
    hero_max_commit: int | None = None,
) -> ClickPlan:
    """Compute the click plan for the bot's intended action.

    Args:
        client_action: dict with keys "kind" ("fold"/"check"/"call"/
            "raise_to"), "chip_amount" (int or None for non-raises),
            "raw_openspiel_chip_int" (int — audit field, not used here).
        controls: scraper record's `controls` dict. Must contain
            `action_buttons` (list of {label, box, amount?}) and, for
            raises, `bet_input` (dict with `box`).
        extended: Stage-2 executor completion (approved 2026-06-11),
            default OFF = byte-identical to the pre-extended mapping.
            When ON:
              - typed raises gain a "verify" step (read back the
                bet_input box before clicking RAISE/BET);
              - an all-in raise_to (chip_amount >= hero_max_commit)
                with no typed path uses the ALLIN button when present;
              - raise_to with no executable raise UI realizes as CALL
                (facing-all-in call-only UI; conservative under-commit)
                or CHECK, in that order;
              - call with no CALL button realizes as CHECK when present
                (to_call==0 rendering variance).
            Realizations are stamped in ClickPlan.realized_as; a
            realization NEVER commits more chips than the intent except
            the ALLIN-button case, which is gated on the intent itself
            being all-in.
        hero_max_commit: hero stack + hero current bet (the raise-to
            ceiling). Required to recognize all-in intent; extended
            ALLIN mapping is skipped when None.

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
            if extended:
                check_box = _box_for_button("CHECK", action_buttons)
                if check_box is not None:
                    return ClickPlan(
                        "call", chip_amount,
                        [ClickStep("click", "CHECK", box=check_box)],
                        realized_as="check")
                # Call-is-all-in UI: when the call amount >= hero's
                # stack the site renders ALLIN instead of CALL — the
                # ALLIN button IS the call (commits exactly the call).
                allin_box = _box_for_button("ALLIN", action_buttons) \
                    or _box_for_button("ALL-IN", action_buttons) \
                    or _box_for_button("ALL IN", action_buttons)
                if allin_box is not None:
                    return ClickPlan(
                        "call", chip_amount,
                        [ClickStep("click", "ALLIN", box=allin_box)],
                        realized_as="allin_button")
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
        typed_path_ok = (bet_input_box_raw is not None
                         and len(bet_input_box_raw) == 4
                         and raise_box is not None)
        if not typed_path_ok and extended:
            # Realization ladder for raise intent with no typed-raise UI.
            # 1. All-in intent + ALLIN button: the literal action.
            is_allin_intent = (hero_max_commit is not None
                               and int(chip_amount) >= int(hero_max_commit))
            allin_box = _box_for_button("ALLIN", action_buttons) \
                or _box_for_button("ALL-IN", action_buttons) \
                or _box_for_button("ALL IN", action_buttons)
            if is_allin_intent and allin_box is not None:
                return ClickPlan(
                    "raise_to", chip_amount,
                    [ClickStep("click", "ALLIN", box=allin_box)],
                    realized_as="allin_button")
            # 2. CALL: the table offers no raise (facing all-in /
            #    mid-render). Conservative under-commit of the intent.
            call_box = _box_for_button("CALL", action_buttons)
            if call_box is not None:
                return ClickPlan(
                    "raise_to", chip_amount,
                    [ClickStep("click", "CALL", box=call_box)],
                    realized_as="call")
            # 3. CHECK: free continuation.
            check_box = _box_for_button("CHECK", action_buttons)
            if check_box is not None:
                return ClickPlan(
                    "raise_to", chip_amount,
                    [ClickStep("click", "CHECK", box=check_box)],
                    realized_as="check")
            # fall through to the default-mode no-op reasons
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
        steps = [
            ClickStep("click", "bet_input", box=bet_input_box),
            ClickStep("type", "bet_input", payload=str(int(chip_amount))),
        ]
        if extended:
            # Stage-2 typed-amount verification: executor must read the
            # box back and abort on mismatch before committing the click.
            steps.append(ClickStep("verify", "bet_input",
                                   payload=str(int(chip_amount))))
        steps.append(ClickStep("click", raise_label, box=raise_box))
        return ClickPlan("raise_to", chip_amount, steps)

    return ClickPlan(kind or "unknown", chip_amount, [],
                      reason=f"unsupported action kind: {kind!r}")


def click_plan_for_safe_fold(reason: str) -> ClickPlan:
    """The no-op plan returned when a frame is safe-folded (invariant
    fail, replay error, missing tracker pre-hand, etc.). The Stage 1
    dashboard shows this with the SAFE-FOLD banner."""
    return ClickPlan("no_plan", None, [], reason=reason)


def plan_as_dict(plan: ClickPlan) -> dict:
    """Serializable form (for JSONL logging). `realized_as` is included
    only when set, so default-mode output is byte-identical to the
    pre-extended serialization."""
    d = {
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
    if plan.realized_as is not None:
        d["realized_as"] = plan.realized_as
    return d
