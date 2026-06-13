"""Extended click-plan mapping (Stage-2 executor completion, approved
2026-06-11). Default mode must stay byte-identical; extended mode adds
the verify step, ALLIN-button mapping, and conservative realizations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nlhe.integration.click_target import (
    compute_click_target, plan_as_dict,
)

BOX = [0.1, 0.2, 0.3, 0.4]


def controls(buttons, bet_input=True):
    return {
        "action_buttons": [{"label": b, "box": BOX} for b in buttons],
        "bet_input": {"box": [0.7, 0.87, 0.81, 0.92]} if bet_input else {},
    }


def raise_to(amount):
    return {"kind": "raise_to", "chip_amount": amount,
            "raw_openspiel_chip_int": amount}


def test_default_mode_unchanged_no_verify_no_realization():
    p = compute_click_target(raise_to(500),
                             controls(["CALL", "FOLD", "RAISE"]))
    assert [s.kind for s in p.steps] == ["click", "type", "click"]
    assert "realized_as" not in plan_as_dict(p)
    # call-only UI still produces a no-op in default mode
    p2 = compute_click_target(raise_to(500), controls(["CALL", "FOLD"]))
    assert p2.is_no_op() and "neither RAISE nor BET" in p2.reason


def test_extended_typed_raise_gains_verify_step():
    p = compute_click_target(raise_to(500),
                             controls(["CALL", "FOLD", "RAISE"]),
                             extended=True, hero_max_commit=2000)
    assert [s.kind for s in p.steps] == ["click", "type", "verify", "click"]
    assert p.steps[2].payload == "500"
    assert p.realized_as is None


def test_extended_call_only_ui_realizes_as_call():
    p = compute_click_target(raise_to(3007), controls(["CALL", "FOLD"]),
                             extended=True, hero_max_commit=3007)
    assert [(s.kind, s.target) for s in p.steps] == [("click", "CALL")]
    assert p.realized_as == "call"
    assert p.action_kind == "raise_to"
    assert plan_as_dict(p)["realized_as"] == "call"


def test_extended_allin_intent_uses_allin_button():
    p = compute_click_target(raise_to(3600),
                             controls(["ALLIN", "CALL", "FOLD"]),
                             extended=True, hero_max_commit=3600)
    assert [(s.kind, s.target) for s in p.steps] == [("click", "ALLIN")]
    assert p.realized_as == "allin_button"


def test_extended_non_allin_intent_never_clicks_allin():
    # A 6BB-raise intent at an ALLIN/CALL/FOLD UI must realize as CALL
    # (under-commit), never as the larger ALLIN.
    p = compute_click_target(raise_to(300),
                             controls(["ALLIN", "CALL", "FOLD"]),
                             extended=True, hero_max_commit=3600)
    assert p.realized_as == "call"


def test_extended_allin_skipped_without_hero_max_commit():
    p = compute_click_target(raise_to(3600),
                             controls(["ALLIN", "CALL", "FOLD"]),
                             extended=True, hero_max_commit=None)
    assert p.realized_as == "call"   # ladder falls through to CALL


def test_extended_raise_never_realizes_as_check():
    # BET present but no bet_input box -> typed path unavailable ->
    # ladder: no ALLIN, no CALL -> withheld (NOT check). A no-bet UI
    # always permits a bet, so this is a read failure, not a real
    # check-only spot.
    p = compute_click_target(raise_to(300), controls(["BET", "CHECK"],
                                                     bet_input=False),
                             extended=True, hero_max_commit=2000)
    assert p.is_no_op()
    assert p.realized_as is None
    assert "bet_input box missing" in p.reason
    # call -> check realization is unaffected (to_call==0 equivalence).
    pc = compute_click_target({"kind": "call", "chip_amount": None},
                              controls(["CHECK", "FOLD"]),
                              extended=True)
    assert pc.realized_as == "check"
    assert pc.action_kind == "call"


# --- Exhibit fixtures: the two live RAISE->CHECK mismatch frames -------------
# Controls blocks verbatim from the live raw_records. Both are pre-action
# panel / mid-render reads: CHECK+FOLD buttons, no RAISE/BET, bet_input
# layout box present with no value. The old ladder emitted a CHECK click
# for a raise decision; the plan must be withheld instead.

# logs/live_dryrun_20260612_230149.jsonl seq=194 — RAISE_TO 610, no-bet
# river, hero stack 1295 + bet 0.
EXHIBIT_SEQ194_CONTROLS = {
    "present": True, "stable": True,
    "action_buttons": [
        {"label": "CHECK", "slot": "center",
         "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
        {"label": "FOLD", "slot": "left",
         "box": [0.366, 0.775, 0.505, 0.857], "amount": None},
    ],
    "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": None},
    "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": False},
}

# logs/live_dryrun_20260611_222532.jsonl seq=2626 — RAISE_TO 716, no-bet
# turn, hero stack 824 + bet 0. Raw text was the "what will you do next
# turn?" pre-action checkbox panel.
EXHIBIT_SEQ2626_CONTROLS = {
    "present": True, "stable": True,
    "action_buttons": [
        {"label": "CHECK", "slot": "center",
         "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
        {"label": "FOLD", "slot": "left",
         "box": [0.366, 0.775, 0.505, 0.857], "amount": None},
    ],
    "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": None},
    "slider": {"box": [0.43, 0.886, 0.672, 0.916], "present": False},
}


def test_exhibit_seq194_raise610_checkfold_panel_withheld():
    p = compute_click_target(
        {"kind": "raise_to", "chip_amount": 610,
         "raw_openspiel_chip_int": 610},
        EXHIBIT_SEQ194_CONTROLS, extended=True, hero_max_commit=1295)
    assert p.is_no_op()
    assert p.realized_as is None
    assert "neither RAISE nor BET" in p.reason
    assert plan_as_dict(p)["steps"] == []
    assert "realized_as" not in plan_as_dict(p)


def test_exhibit_seq2626_raise716_checkfold_panel_withheld():
    p = compute_click_target(
        {"kind": "raise_to", "chip_amount": 716,
         "raw_openspiel_chip_int": 716},
        EXHIBIT_SEQ2626_CONTROLS, extended=True, hero_max_commit=824)
    assert p.is_no_op()
    assert p.realized_as is None
    assert "neither RAISE nor BET" in p.reason


def test_exhibit_seq2629_allin_check_ui_non_allin_intent_withheld():
    # logs/live_dryrun_20260611_222532.jsonl seq=2629 sub-case: right
    # slot reads ALLIN(824) because the bet_input holds the full stack;
    # intent 716 < 824 is NOT all-in, no CALL -> withheld (old code
    # clicked CHECK here too).
    ctl = {
        "action_buttons": [
            {"label": "ALLIN", "slot": "right",
             "box": [0.668, 0.775, 0.8, 0.857], "amount": 824},
            {"label": "CHECK", "slot": "center",
             "box": [0.515, 0.775, 0.66, 0.857], "amount": None},
        ],
        "bet_input": {"box": [0.705, 0.876, 0.812, 0.922], "value": 824},
    }
    p = compute_click_target(
        {"kind": "raise_to", "chip_amount": 716,
         "raw_openspiel_chip_int": 716},
        ctl, extended=True, hero_max_commit=824)
    assert p.is_no_op()
    assert p.realized_as is None


def test_extended_no_ui_still_no_op():
    p = compute_click_target(raise_to(500), controls(["FOLD"]),
                             extended=True, hero_max_commit=2000)
    assert p.is_no_op()


def test_extended_call_is_allin_ui_realizes_via_allin_button():
    # Site renders ALLIN instead of CALL when calling commits the whole
    # stack (live seq 994): the ALLIN button IS the call.
    p = compute_click_target({"kind": "call", "chip_amount": None},
                             controls(["ALLIN", "FOLD"]), extended=True)
    assert [(s.kind, s.target) for s in p.steps] == [("click", "ALLIN")]
    assert p.realized_as == "allin_button"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
