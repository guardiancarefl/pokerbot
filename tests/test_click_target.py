"""Unit tests for src/nlhe/integration/click_target.py."""
from __future__ import annotations

import pytest

from src.nlhe.integration.click_target import (
    ClickPlan,
    ClickStep,
    click_plan_for_safe_fold,
    compute_click_target,
    plan_as_dict,
)


def _controls(buttons, bet_input_box=None):
    """Build a minimal scraper controls dict for tests."""
    out = {"action_buttons": buttons}
    if bet_input_box is not None:
        out["bet_input"] = {"box": bet_input_box, "value": None}
    return out


FOLD_BTN = {"label": "FOLD", "slot": "left",
            "box": [0.366, 0.775, 0.505, 0.857], "amount": None}
CHECK_BTN = {"label": "CHECK", "slot": "center",
             "box": [0.515, 0.775, 0.66, 0.857], "amount": None}
CALL_BTN = {"label": "CALL", "slot": "center",
            "box": [0.515, 0.775, 0.66, 0.857], "amount": 235}
RAISE_BTN = {"label": "RAISE", "slot": "right",
             "box": [0.668, 0.775, 0.8, 0.857], "amount": 535}
BET_INPUT_BOX = [0.705, 0.876, 0.812, 0.922]


def test_fold_plan_single_click():
    plan = compute_click_target(
        {"kind": "fold", "chip_amount": None, "raw_openspiel_chip_int": 0},
        _controls([FOLD_BTN, CALL_BTN, RAISE_BTN]),
    )
    assert plan.action_kind == "fold"
    assert plan.chip_amount is None
    assert plan.reason is None
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.kind == "click"
    assert step.target == "FOLD"
    assert step.box == (0.366, 0.775, 0.505, 0.857)


def test_check_plan_single_click():
    plan = compute_click_target(
        {"kind": "check", "chip_amount": None, "raw_openspiel_chip_int": 1},
        _controls([CHECK_BTN, {"label": "BET", "box": [0, 0, 0, 0]}]),
    )
    assert plan.action_kind == "check"
    assert len(plan.steps) == 1
    assert plan.steps[0].target == "CHECK"


def test_call_plan_carries_chip_amount():
    plan = compute_click_target(
        {"kind": "call", "chip_amount": 235,
         "raw_openspiel_chip_int": 1},
        _controls([CALL_BTN, FOLD_BTN, RAISE_BTN]),
    )
    assert plan.action_kind == "call"
    assert plan.chip_amount == 235
    assert len(plan.steps) == 1
    assert plan.steps[0].target == "CALL"


def test_raise_plan_three_steps():
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 1062,
         "raw_openspiel_chip_int": 1062},
        _controls([CALL_BTN, FOLD_BTN, RAISE_BTN],
                   bet_input_box=BET_INPUT_BOX),
    )
    assert plan.action_kind == "raise_to"
    assert plan.chip_amount == 1062
    assert plan.reason is None
    assert len(plan.steps) == 3

    step1, step2, step3 = plan.steps
    assert step1.kind == "click"
    assert step1.target == "bet_input"
    assert step1.box == tuple(BET_INPUT_BOX)
    assert step2.kind == "type"
    assert step2.target == "bet_input"
    assert step2.payload == "1062"
    assert step3.kind == "click"
    assert step3.target == "RAISE"
    assert step3.box == (0.668, 0.775, 0.8, 0.857)


def test_safe_fold_when_fold_button_missing():
    """Read issue: scraper didn't see the FOLD button. Plan is no-op
    with an explicit reason."""
    plan = compute_click_target(
        {"kind": "fold", "chip_amount": None, "raw_openspiel_chip_int": 0},
        _controls([CALL_BTN, RAISE_BTN]),  # no FOLD
    )
    assert plan.action_kind == "fold"
    assert plan.steps == []
    assert plan.reason is not None
    assert "FOLD" in plan.reason


def test_raise_safe_folds_when_bet_input_missing():
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 100,
         "raw_openspiel_chip_int": 100},
        _controls([CALL_BTN, FOLD_BTN, RAISE_BTN]),  # no bet_input
    )
    assert plan.action_kind == "raise_to"
    assert plan.steps == []
    assert plan.reason is not None
    assert "bet_input" in plan.reason


def test_raise_safe_folds_when_raise_and_bet_buttons_missing():
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 100,
         "raw_openspiel_chip_int": 100},
        _controls([CALL_BTN, FOLD_BTN], bet_input_box=BET_INPUT_BOX),
    )
    assert plan.action_kind == "raise_to"
    assert plan.steps == []
    assert plan.reason is not None
    assert "RAISE" in plan.reason or "BET" in plan.reason


BET_BTN = {"label": "BET", "slot": "right",
            "box": [0.668, 0.775, 0.8, 0.857], "amount": 50}
POSTFLOP_CHECK_BTN = {"label": "CHECK", "slot": "center",
                       "box": [0.515, 0.775, 0.66, 0.857], "amount": None}


def test_raise_to_uses_bet_button_when_raise_not_present():
    """Postflop hero opening with no bet pending: the action_button is
    labeled BET, not RAISE. The bot's `raise_to` action_kind still
    applies (one OpenSpiel chip_int → either label depending on street).
    Plan should resolve cleanly via the BET button."""
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 50,
         "raw_openspiel_chip_int": 50},
        _controls([BET_BTN, POSTFLOP_CHECK_BTN],
                   bet_input_box=BET_INPUT_BOX),
    )
    assert plan.action_kind == "raise_to"
    assert plan.chip_amount == 50
    assert plan.reason is None
    assert len(plan.steps) == 3
    # The third step clicks the BET button (the label that was actually present)
    assert plan.steps[2].target == "BET"
    assert plan.steps[2].box == (0.668, 0.775, 0.8, 0.857)


def test_raise_prefers_raise_when_both_present():
    """If both RAISE and BET appear (unusual but possible during UI
    transitions), prefer RAISE — it's the canonical name on streets
    where the table has both."""
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 100,
         "raw_openspiel_chip_int": 100},
        _controls([RAISE_BTN, BET_BTN, FOLD_BTN],
                   bet_input_box=BET_INPUT_BOX),
    )
    assert plan.steps[2].target == "RAISE"


def test_raise_with_no_chip_amount_safe_folds():
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": None,
         "raw_openspiel_chip_int": 100},
        _controls([CALL_BTN, FOLD_BTN, RAISE_BTN],
                   bet_input_box=BET_INPUT_BOX),
    )
    assert plan.steps == []
    assert "chip_amount" in plan.reason


def test_unknown_kind_safe_folds():
    plan = compute_click_target(
        {"kind": "muck", "chip_amount": None,
         "raw_openspiel_chip_int": 0},
        _controls([FOLD_BTN]),
    )
    assert plan.steps == []
    assert "unsupported" in plan.reason


def test_label_matching_is_case_insensitive():
    plan = compute_click_target(
        {"kind": "fold", "chip_amount": None, "raw_openspiel_chip_int": 0},
        _controls([{"label": "fold", "box": [0.1, 0.2, 0.3, 0.4],
                     "slot": "left", "amount": None}]),
    )
    assert plan.steps and plan.steps[0].box == (0.1, 0.2, 0.3, 0.4)


def test_safe_fold_helper_produces_no_op():
    plan = click_plan_for_safe_fold("invariant_fail: stack[seat3] off by 5")
    assert plan.action_kind == "no_plan"
    assert plan.steps == []
    assert plan.reason == "invariant_fail: stack[seat3] off by 5"
    assert plan.is_no_op()


def test_plan_as_dict_serializable():
    plan = compute_click_target(
        {"kind": "raise_to", "chip_amount": 50,
         "raw_openspiel_chip_int": 50},
        _controls([CALL_BTN, FOLD_BTN, RAISE_BTN],
                   bet_input_box=BET_INPUT_BOX),
    )
    d = plan_as_dict(plan)
    # Must be JSON-encodable
    import json
    json.dumps(d)  # raises if not
    assert d["action_kind"] == "raise_to"
    assert d["chip_amount"] == 50
    assert d["steps"][1]["payload"] == "50"
    assert d["steps"][2]["box"] == list(RAISE_BTN["box"])


def test_controls_none_safe_folds_cleanly():
    plan = compute_click_target(
        {"kind": "fold", "chip_amount": None, "raw_openspiel_chip_int": 0},
        None,
    )
    assert plan.steps == []
    assert plan.reason is not None
