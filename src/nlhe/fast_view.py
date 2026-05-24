"""Fast view + discretization for the rollout hot loop (B1c sub-step 2, Stage A).

Under `bettingAbstraction=fullgame`, `state.legal_actions()` returns ~9,803
chip-action ints. The canonical `cfr6._build_view_6max` builds a `set()` and a
list-comprehension over that whole list (~0.64 ms/step), and
`actions.discretize_legal_actions` builds another `set()` (~0.10-0.24 ms). In a
leaf-eval rollout these run at every decision step, making them the binding cost
of leaf evaluation — NOT the regex parse (`parse_state_6max` is 0.008 ms; see
`docs/SUBGAME_LEAF_DESIGN.md` Q4 for the corrected decomposition).

`state.legal_actions()` is **sorted ascending** (`[0, 1, 200, 201, ..., 10000]`).
This module exploits that to derive the SAME `GameStateView` and discretization
via `bisect` (O(log n) membership, O(1) min/max), with **field-by-field-identical
output** to the canonical path and no 9,803-element `set`/list-comp/min/max.

Scope: this is a PARALLEL module consumed only by the leaf evaluator. The
canonical `_build_view_6max` / `discretize_legal_actions` are untouched. Folding
this into the canonical path (which benefits the CFR walker and every other
consumer, including training) is a tracked follow-up — see `NEXT_SESSION.md`.

Contract: **decision nodes only.** `fast_view_and_discretize` raises on chance
and terminal states. This mirrors `cfr6.traverse_6max`, which never calls
`_build_view_6max` on a chance or terminal node — chance is handled by sampling
`state.chance_outcomes()` and terminals stop the rollout before any view is built.
"""
from __future__ import annotations

import bisect
from typing import Any

from src.nlhe.actions import (
    DiscreteAction,
    GameStateView,
    _legal_discrete_bet_sizes,
)


def _contains_sorted(sorted_seq, x) -> bool:
    """Membership test on an ascending-sorted sequence via bisect (O(log n)).

    Equivalent to ``x in set(sorted_seq)`` but without building the set.
    """
    i = bisect.bisect_left(sorted_seq, x)
    return i < len(sorted_seq) and sorted_seq[i] == x


def fast_build_view(parsed: dict, legal_sorted) -> GameStateView:
    """`GameStateView` identical to `cfr6._build_view_6max(state, parsed)`.

    The pot / to_call / effective_stack math is copied verbatim from the
    canonical path (it is O(num_players), not a bottleneck). Only the legal-action
    handling is optimized: `min_bet` / `max_bet` / `legal_fold` / `legal_call`
    come from bisect on the SORTED `legal_sorted` instead of `set(legal)`,
    `[a for a in legal if a >= 2]`, and `min`/`max` over ~9,803 elements.

    Args:
        parsed: a `parse_state_6max(state)` dict (same object the canonical path
            receives). Provides current_player / money / contribution / pot.
        legal_sorted: `state.legal_actions()` — sorted ascending under
            universal_poker. NOT copied; read-only here.

    Returns:
        GameStateView, byte-for-byte equal to the canonical builder's output.
    """
    cp = parsed["current_player"]
    money = parsed["money"]
    contribution = parsed["contribution"]
    pot = parsed["pot"]

    # --- verbatim from _build_view_6max (cheap O(num_players) work) ---
    my_money = money[cp] if cp < len(money) else 0
    max_contrib = max(contribution) if contribution else 0
    my_contrib = contribution[cp] if cp < len(contribution) else 0
    to_call = max(0, max_contrib - my_contrib)

    opp_stacks = [money[i] for i in range(len(money))
                  if i != cp and money[i] > 0]
    if opp_stacks:
        effective_stack = min(my_money, max(opp_stacks))
    else:
        effective_stack = my_money

    # --- optimized part: sorted-list inspection, not set/list-comp/min/max ---
    # OpenSpiel universal_poker: 0=fold, 1=call/check, N>=2=bet-to-N-chips.
    # bet actions are everything >= 2; since legal_sorted is ascending they form a
    # contiguous suffix. first_bet_idx is the index of the smallest bet action.
    n = len(legal_sorted)
    first_bet_idx = bisect.bisect_left(legal_sorted, 2)
    if first_bet_idx < n:
        min_bet = legal_sorted[first_bet_idx]   # == min(a for a in legal if a>=2)
        max_bet = legal_sorted[n - 1]           # == max(...) (largest is a bet)
    else:
        min_bet = 0
        max_bet = 0

    return GameStateView(
        pot=pot,
        to_call=to_call,
        effective_stack=effective_stack,
        min_bet=min_bet,
        max_bet=max_bet,
        legal_fold=_contains_sorted(legal_sorted, 0),
        legal_call=_contains_sorted(legal_sorted, 1),
    )


def fast_discretize(legal_sorted, view: GameStateView) -> dict:
    """Identical output to `actions.discretize_legal_actions(legal_sorted, view)`.

    Membership tests use bisect on the sorted list instead of building a
    ~9,803-element `set`. `_legal_discrete_bet_sizes(view)` is reused unchanged
    (it reads only the view, not the legal list, and is cheap), so the candidate
    bet sizes and their dedup/ordering are exactly the canonical ones.

    Returns:
        {DiscreteAction: chip_action} — same keys/values/insertion-order as the
        canonical discretizer.
    """
    out: dict[DiscreteAction, int] = {}
    if _contains_sorted(legal_sorted, 0):
        out[DiscreteAction.FOLD] = 0
    if _contains_sorted(legal_sorted, 1):
        out[DiscreteAction.CALL] = 1
    for action, chips in _legal_discrete_bet_sizes(view):
        if _contains_sorted(legal_sorted, chips):
            out[action] = chips
    return out


def fast_view_and_discretize(state: Any, parsed: dict):
    """Hot-loop entry point: ONE `legal_actions()` call shared by view + discretize.

    This is the function the rollout calls at each DECISION step. It fetches
    `state.legal_actions()` exactly once and threads it through both
    `fast_build_view` and `fast_discretize`, eliminating the duplicate call the
    canonical path makes (view fetches it, then discretize is handed
    `list(state.legal_actions())` separately).

    Decision nodes only. Raises `ValueError` on chance and terminal states —
    rollout code samples chance via `state.chance_outcomes()` and stops at
    terminals before reaching here (mirroring `cfr6.traverse_6max`, which never
    builds a view on a chance/terminal node).

    Args:
        state: an OpenSpiel universal_poker decision state.
        parsed: `parse_state_6max(state)` (the caller already needs it for the
            encoder, so it is passed in rather than re-parsed here).

    Returns:
        (view, discrete_to_chip, legal_sorted):
          - view: GameStateView (== canonical `_build_view_6max`)
          - discrete_to_chip: {DiscreteAction: chip} (== canonical discretizer)
          - legal_sorted: the shared `state.legal_actions()` list (sorted asc),
            returned so the caller can reuse it without a second native call.

    Raises:
        ValueError: if `state` is a chance node or terminal.
    """
    if state.is_chance_node():
        raise ValueError(
            "fast_view_and_discretize: chance node has no decision view — "
            "rollout must sample chance via state.chance_outcomes()"
        )
    if state.is_terminal():
        raise ValueError(
            "fast_view_and_discretize: terminal state has no decision view"
        )
    legal_sorted = state.legal_actions()  # sorted ascending; one call, shared
    view = fast_build_view(parsed, legal_sorted)
    discrete_to_chip = fast_discretize(legal_sorted, view)
    return view, discrete_to_chip, legal_sorted
