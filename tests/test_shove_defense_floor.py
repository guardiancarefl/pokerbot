"""Unit tests for the slate-2a shove-defense floor wired into live_loop.

Covers (pre-registered, evals/shove_floor_wiring_20260613):
  - nearest-cell mapping (depth + level grids, tie->lower; shover SB/UTG).
  - fires on a weak (oracle=fold) facing-all-in 5-15bb hand: CALL/ALLIN mass
    moves to FOLD.
  - does NOT fire on a strong (oracle=call) hand: identity short-circuit.
  - does NOT fire off-scope (not facing-all-in / not preflop / deep stack).
  - composition with the tail floor through make_live_policy_filter: both
    can fire on the same decision.
  - range-table-missing / empty-ranges refusal (load_shove_defense_floor).

The fire/no-fire cases build REAL OpenSpiel facing-shove states via the
battery's build_facing_shove_spot, then parse with parse_state_6max — the
same parse the live path feeds the filter — so the test exercises the real
qualify + ICM + equity path, not a synthetic parsed dict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.integration.live_loop import (
    ShoveDefenseFloor,
    load_shove_defense_floor,
    make_live_policy_filter,
    _shove_nearest_grid,
    _shove_qualify,
    _SHOVE_DEPTH_GRID,
    _SHOVE_LEVEL_GRID,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY = REPO_ROOT / "evals/h2_battery/battery_v1.json"

N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)
A_BET_33 = int(DiscreteAction.BET_33)
A_BET_100 = int(DiscreteAction.BET_100)


# --------------------------------------------------------------------------
# Shared fixtures: a real loaded floor + a real-ante structure.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def battery():
    return json.loads(BATTERY.read_text())


@pytest.fixture(scope="module")
def structure(battery):
    from scripts.throwaway_query_real_ante import _RealAnteStructure
    return _RealAnteStructure(
        TournamentStructure.from_yaml(battery["structure"]))


@pytest.fixture(scope="module")
def floor(structure):
    return load_shove_defense_floor(
        range_table_path=str(BATTERY), tau=0.0, structure=structure)


def _build_spot(structure, s):
    """Build a real facing-shove OpenSpiel state + parse it as the live
    path does (observer = HERO_SEAT)."""
    from scripts.fold_vs_shove_battery import build_facing_shove_spot
    from scripts.depth_invariance_probe import HERO_SEAT
    from src.nlhe.infoset6 import parse_state_6max
    state, _dealer, _bb, _sh = build_facing_shove_spot(
        structure, s["level"], s["hero_pos"], s["shover_pos"],
        tuple(s["hero_cards"]), s["depth_bb"])
    parsed = parse_state_6max(state, observer=HERO_SEAT)
    return state, parsed


def _facing_shove_policy_mask():
    """A policy that puts real mass on CALL/ALLIN (the actions the floor
    moves to FOLD) and FOLD, with only those three legal."""
    p = np.zeros(N_ACT, dtype=np.float32)
    p[A_FOLD] = 0.30
    p[A_CALL] = 0.30
    p[A_ALLIN] = 0.40
    mask = np.zeros(N_ACT, dtype=np.float32)
    mask[A_FOLD] = mask[A_CALL] = mask[A_ALLIN] = 1.0
    return p, mask


# --------------------------------------------------------------------------
# Nearest-cell mapping (no state needed).
# --------------------------------------------------------------------------

def test_nearest_grid_depth_exact():
    for g in _SHOVE_DEPTH_GRID:
        assert _shove_nearest_grid(float(g), _SHOVE_DEPTH_GRID) == g


def test_nearest_grid_depth_offgrid():
    # 6 -> 5 (closer), 7 -> 8 (closer), 9 -> 8, 13 -> 11(|2| vs |2|, tie->lower)
    assert _shove_nearest_grid(6.0, _SHOVE_DEPTH_GRID) == 5
    assert _shove_nearest_grid(7.0, _SHOVE_DEPTH_GRID) == 8
    assert _shove_nearest_grid(9.0, _SHOVE_DEPTH_GRID) == 8
    # 13 is equidistant to 11 and 15 -> tie resolves LOWER
    assert _shove_nearest_grid(13.0, _SHOVE_DEPTH_GRID) == 11


def test_nearest_grid_level_tie_lower():
    # level 4 is equidistant to 3 and 5 -> tie -> 3
    assert _shove_nearest_grid(4.0, _SHOVE_LEVEL_GRID) == 3
    # level 6 equidistant 5/7 -> 5
    assert _shove_nearest_grid(6.0, _SHOVE_LEVEL_GRID) == 5
    # below grid maps to lowest, above grid to highest
    assert _shove_nearest_grid(1.0, _SHOVE_LEVEL_GRID) == 3
    assert _shove_nearest_grid(9.0, _SHOVE_LEVEL_GRID) == 7


def test_qualify_cell_key_matches_grid(structure, floor, battery):
    """A constructed UTG-shover 5bb/L3 spot maps to cell UTG|5|3."""
    s = next(sp for sp in battery["spots"]
             if sp["shover_pos"] == "UTG" and sp["depth_bb"] == 5
             and sp["level"] == 3)
    _state, parsed = _build_spot(structure, s)
    q = _shove_qualify(parsed, _state, floor.bb_to_level)
    assert q is not None
    cell_key, _hero_cards, _ef, _ew, _el, depth_bb = q
    assert cell_key == "UTG|5|3"
    assert abs(depth_bb - 5.0) < 1e-6


def test_qualify_shover_sb_cell(structure, floor, battery):
    """A BB-vs-SB shover spot maps to an SB|... cell (shover posted a blind)."""
    s = next((sp for sp in battery["spots"] if sp["shover_pos"] == "SB"), None)
    assert s is not None
    _state, parsed = _build_spot(structure, s)
    q = _shove_qualify(parsed, _state, floor.bb_to_level)
    assert q is not None
    assert q[0].startswith("SB|")


# --------------------------------------------------------------------------
# Fires / doesn't fire.
# --------------------------------------------------------------------------

def test_fires_on_weak_fold_hand(structure, floor, battery):
    """An oracle=fold spot below break-even: CALL/ALLIN mass -> FOLD."""
    s = next(sp for sp in battery["spots"]
             if sp["shover_pos"] == "UTG" and sp["depth_bb"] == 5
             and sp["level"] == 3 and sp["oracle"] == "fold")
    state, parsed = _build_spot(structure, s)
    p, mask = _facing_shove_policy_mask()
    out = floor.apply(p, mask, parsed, state)
    assert out is not p                      # fired
    assert out[A_FOLD] == 1.0
    assert out[A_CALL] == 0.0
    assert out[A_ALLIN] == 0.0


def test_no_fire_on_strong_call_hand(structure, battery):
    """An oracle=call spot (e.g. AA) clears the break-even gate: identity."""
    fl = load_shove_defense_floor(range_table_path=str(BATTERY), tau=0.0,
                                  structure=structure)
    s = next(sp for sp in battery["spots"]
             if sp["shover_pos"] == "UTG" and sp["depth_bb"] == 5
             and sp["level"] == 3 and sp["oracle"] == "call")
    state, parsed = _build_spot(structure, s)
    p, mask = _facing_shove_policy_mask()
    out = fl.apply(p, mask, parsed, state)
    assert out is p                          # identity short-circuit
    assert fl.stats["n_qualify"] >= 1        # it DID qualify, just cleared


def test_no_fire_off_scope_deep_stack(structure, floor):
    """A deep (30bb) preflop spot — not facing all-in — never qualifies."""
    # Synthetic parsed dict; state only needed for _shove_blind_info, which
    # short-circuits the to_call guard before touching state. Use a real
    # state's params is unnecessary here because the to_call/commit guard
    # fails first. Provide a minimal state stub via a real spot would be
    # heavier; instead assert via _shove_qualify on a non-facing dict.
    parsed = {
        "street_idx": 0, "big_blind": 100, "current_player": 0,
        "money": [3000, 3000, 3000, 3000, 3000, 3000],
        "contribution": [50, 100, 0, 0, 0, 0],   # to_call=50 << stack
        "private_cards": "7d2c",
    }

    class _StubState:
        def get_game(self):
            class _G:
                def get_parameters(self_inner):
                    return {"blind": "0 50 100 0 0 0", "ante": "0 0 0 0 0 0"}
            return _G()

    q = _shove_qualify(parsed, _StubState(), floor.bb_to_level)
    assert q is None                         # to_call < stack -> no commit


def test_no_fire_postflop(floor):
    """Postflop (street_idx != 0) never qualifies."""
    parsed = {"street_idx": 1, "big_blind": 100, "current_player": 0,
              "money": [0, 0, 0, 0, 0, 0], "contribution": [500] * 6,
              "private_cards": "7d2c"}
    q = _shove_qualify(parsed, None, floor.bb_to_level)
    assert q is None


# --------------------------------------------------------------------------
# Composition with the tail floor through make_live_policy_filter.
# --------------------------------------------------------------------------

def test_composes_with_tail_floor(structure, floor, battery):
    """On a weak facing-all-in spot, the chain with BOTH shove-defense and
    the tail floor armed fires (the composed filter routes shove-defense
    after short-stack and before the tail floor; both may fire)."""
    s = next(sp for sp in battery["spots"]
             if sp["shover_pos"] == "UTG" and sp["depth_bb"] == 5
             and sp["level"] == 3 and sp["oracle"] == "fold")
    state, parsed = _build_spot(structure, s)
    p, mask = _facing_shove_policy_mask()

    filt = make_live_policy_filter(
        short_stack_threshold_bb=6.0, tail_floor_tau=0.10,
        shove_defense_floor=floor)
    out = filt(p, mask, parsed, state)
    # Shove-defense alone already collapses to FOLD=1.0; the tail floor
    # cannot prune FOLD (commit 0). End state: pure FOLD.
    assert out is not p
    assert out[A_FOLD] == 1.0


def test_off_chain_identity_when_floor_none(structure, battery):
    """make_live_policy_filter with shove_defense_floor=None never touches
    a weak facing-all-in spot via the shove path (off-chain identity)."""
    s = next(sp for sp in battery["spots"]
             if sp["shover_pos"] == "UTG" and sp["depth_bb"] == 5
             and sp["level"] == 3 and sp["oracle"] == "fold")
    state, parsed = _build_spot(structure, s)
    # Use a non-short stack so short-stack floor also can't fire: depth here
    # is 5bb so short-stack WOULD fire; isolate by giving the deep-stack
    # synthetic policy with only FOLD/CALL/ALLIN legal (short-stack keeps
    # exactly those, so it is a structural no-op). The point: no shove fold.
    p, mask = _facing_shove_policy_mask()
    filt = make_live_policy_filter(short_stack_threshold_bb=6.0,
                                   shove_defense_floor=None)
    out = filt(p, mask, parsed, state)
    # The short-stack floor fires at 4.85bb (it keeps {FOLD,CALL,ALLIN} and
    # renormalizes), but the SHOVE-DEFENSE path is off: FOLD must NOT be
    # collapsed to 1.0 and CALL/ALLIN mass must survive.
    assert out[A_FOLD] < 0.99
    assert out[A_CALL] > 0.0
    assert out[A_ALLIN] > 0.0
    # Same result as the explicit off-floor compose (shove path never ran).
    filt2 = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out2 = filt2(p, mask, parsed, state)
    assert np.allclose(out, out2)


# --------------------------------------------------------------------------
# Range-table refusal.
# --------------------------------------------------------------------------

def test_missing_range_table_refuses(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_shove_defense_floor(
            range_table_path=str(tmp_path / "does_not_exist.json"))


def test_empty_ranges_refuses(tmp_path, structure):
    bad = tmp_path / "no_ranges.json"
    bad.write_text(json.dumps({
        "structure": "configs/ignition_double_up_6max_turbo.yaml",
        "ranges": {}}))
    with pytest.raises(ValueError):
        load_shove_defense_floor(range_table_path=str(bad),
                                 structure=structure)
