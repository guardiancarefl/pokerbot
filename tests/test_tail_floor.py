"""Unit tests for the H1 commitment-scaled tail floor.

Binding spec: docs/research_program/H1_TAIL_FLOOR_SPEC.md.

Verifies:
  - apply_commitment_tail_floor: identity short-circuit, commitment-scaled
    threshold tau(a) = tau_max * commit_frac(a), FOLD/free-CHECK exemption
    (commit 0), big-CALL ~ ALLIN equivalence, renormalization, degenerate
    guard, argmax preservation when argmax mass > tau_max.
  - make_live_policy_filter wiring: tail_floor_tau=None NEVER calls the
    function (structural OFF-path identity); when set, it runs LAST,
    after the short-stack floor.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.nlhe.actions import DiscreteAction
from src.nlhe.integration import live_loop
from src.nlhe.integration.live_loop import (
    apply_commitment_tail_floor,
    make_live_policy_filter,
)


N_ACT = len(DiscreteAction)
A_FOLD = int(DiscreteAction.FOLD)
A_CALL = int(DiscreteAction.CALL)
A_ALLIN = int(DiscreteAction.ALLIN)
A_BET_33 = int(DiscreteAction.BET_33)
A_BET_50 = int(DiscreteAction.BET_50)
A_BET_66 = int(DiscreteAction.BET_66)
A_BET_100 = int(DiscreteAction.BET_100)
A_BET_150 = int(DiscreteAction.BET_150)
A_BET_200 = int(DiscreteAction.BET_200)
BET_SLOTS = (A_BET_33, A_BET_50, A_BET_66, A_BET_100, A_BET_150, A_BET_200)


def _mk_policy(masses):
    """Build a policy from a {action_idx: mass} dict; remaining mass
    spread over unspecified slots."""
    p = np.zeros(N_ACT, dtype=np.float32)
    for idx, m in masses.items():
        p[idx] = m
    rem = 1.0 - float(p.sum())
    assert rem >= -1e-6
    free = [b for b in range(N_ACT) if b not in masses]
    if free and rem > 0:
        for b in free:
            p[b] = rem / len(free)
    assert abs(p.sum() - 1.0) < 1e-5
    return p


def _full_legal():
    return np.ones(N_ACT, dtype=np.float32)


def _parsed_deep_facing():
    """Hero stack 1000, to_call 25, pot 75 (BB=50). commit_frac(ALLIN)=1,
    commit_frac(CALL)=0.025, commit_frac(BET_33)=0.058."""
    return {"current_player": 0,
            "money": [1000, 1500, 1500, 1500, 1500, 1500],
            "contribution": [25, 50, 0, 0, 0, 0],
            "big_blind": 50,
            "public_cards": "", "private_cards": "9h2d", "street_idx": 0}


# --------------------------------------------------------------------------
# apply_commitment_tail_floor — direct unit tests
# --------------------------------------------------------------------------

def test_identity_reference_when_nothing_fires():
    """All masses above their tau → same reference back."""
    p = _mk_policy({A_FOLD: 0.30, A_CALL: 0.30, A_ALLIN: 0.20})
    out = apply_commitment_tail_floor(p, _full_legal(), _parsed_deep_facing(),
                                      state=None, tau_max=0.10)
    assert out is p


def test_allin_75bp_pruned_at_010_and_015_not_005():
    """tau(ALLIN) = tau_max (commit_frac=1). 7.5% ALLIN mass: pruned at
    tau_max 0.10 and 0.15, kept (identity) at 0.05 — the seq-315 class."""
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.075,
                      A_BET_100: 0.125})
    parsed = _parsed_deep_facing()
    for tau in (0.10, 0.15):
        out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                          state=None, tau_max=tau)
        assert out is not p, f"should fire at tau_max={tau}"
        assert out[A_ALLIN] == 0.0
        assert abs(out.sum() - 1.0) < 1e-5
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.05)
    assert out is p, "0.075 >= tau(ALLIN)=0.05 — must not fire"


def test_fold_never_pruned():
    """FOLD commits 0 chips → tau(FOLD)=0 → 0 < mass < 0 is impossible."""
    for fold_mass in (1e-6, 0.001, 0.05):
        p = _mk_policy({A_FOLD: fold_mass, A_CALL: 0.5,
                          A_ALLIN: 0.5 - fold_mass})
        out = apply_commitment_tail_floor(p, _full_legal(),
                                          _parsed_deep_facing(),
                                          state=None, tau_max=0.15)
        assert out[A_FOLD] > 0.0, f"FOLD pruned at mass={fold_mass}"


def test_free_check_never_pruned():
    """to_call == 0 → CALL is a free CHECK, commit 0 → never pruned."""
    p = _mk_policy({A_FOLD: 0.50, A_CALL: 0.001, A_ALLIN: 0.30,
                      A_BET_100: 0.199})
    parsed = {"current_player": 0,
              "money": [1000, 1500, 1500, 1500, 1500, 1500],
              "contribution": [50, 50, 50, 50, 50, 50],   # to_call = 0
              "big_blind": 50}
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.15)
    assert out[A_CALL] > 0.0


def test_small_commit_bet_not_pruned():
    """Commitment scaling: 5% mass on BET_33 (commit_frac=0.058 →
    tau=0.0058 at tau_max=0.10) survives; same 5% on ALLIN would not."""
    p = _mk_policy({A_FOLD: 0.45, A_CALL: 0.45, A_BET_33: 0.05,
                      A_ALLIN: 0.05})
    out = apply_commitment_tail_floor(p, _full_legal(), _parsed_deep_facing(),
                                      state=None, tau_max=0.10)
    assert out is not p
    assert out[A_BET_33] > 0.0, "small-commit bet must survive"
    assert out[A_ALLIN] == 0.0, "same mass on ALLIN must be pruned"


def test_big_call_pruned_like_allin():
    """Calling a shove (to_call ~ stack): commit_frac(CALL) ~ 1 → a 7.5%
    CALL tail is pruned just like a 7.5% ALLIN."""
    parsed = {"current_player": 0,
              "money": [1000, 5, 1500, 1500, 1500, 1500],
              "contribution": [25, 990, 0, 0, 0, 0],  # to_call=965
              "big_blind": 50}
    p = _mk_policy({A_FOLD: 0.85, A_CALL: 0.075, A_ALLIN: 0.075})
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.10)
    assert out is not p
    assert out[A_CALL] == 0.0, "near-full-stack CALL must be pruned"
    assert out[A_ALLIN] == 0.0
    assert abs(out.sum() - 1.0) < 1e-5


def test_renormalization_sums_to_one_over_legal_mask():
    mask = _full_legal()
    mask[A_BET_150] = 0.0
    mask[A_BET_200] = 0.0
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.06,
                      A_BET_33: 0.14, A_BET_150: 0.0, A_BET_200: 0.0})
    out = apply_commitment_tail_floor(p, mask, _parsed_deep_facing(),
                                      state=None, tau_max=0.10)
    assert out is not p
    assert abs(float((out * mask).sum()) - 1.0) < 1e-5
    assert abs(float(out.sum()) - 1.0) < 1e-5
    # Kept actions retain relative shares
    assert abs(out[A_FOLD] / out[A_CALL] - 1.0) < 1e-5


def test_degenerate_guard_returns_input_unchanged():
    """If pruning would zero out everything, return the input reference.
    Only ALLIN legal with full mass; tau_max=2.0 → 1.0 < tau(ALLIN)=2.0."""
    mask = np.zeros(N_ACT, dtype=np.float32)
    mask[A_ALLIN] = 1.0
    p = np.zeros(N_ACT, dtype=np.float32)
    p[A_ALLIN] = 1.0
    out = apply_commitment_tail_floor(p, mask, _parsed_deep_facing(),
                                      state=None, tau_max=2.0)
    assert out is p


def test_noop_when_parsed_fields_missing_or_stack_nonpositive():
    p = _mk_policy({A_FOLD: 0.5, A_ALLIN: 0.01, A_CALL: 0.49})
    # missing money
    out = apply_commitment_tail_floor(
        p, _full_legal(),
        {"current_player": 0, "contribution": [25, 50, 0, 0, 0, 0]},
        state=None, tau_max=0.15)
    assert out is p
    # missing contribution
    out = apply_commitment_tail_floor(
        p, _full_legal(), {"current_player": 0, "money": [1000] * 6},
        state=None, tau_max=0.15)
    assert out is p
    # stack <= 0
    parsed = _parsed_deep_facing()
    parsed["money"] = [0, 1500, 1500, 1500, 1500, 1500]
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.15)
    assert out is p


def test_argmax_unchanged_when_argmax_mass_above_tau_max():
    """tau(a) <= tau_max for every action, so an argmax with mass >
    tau_max can never be pruned; proportional renormalization keeps it
    the argmax. Property-checked over random policies."""
    rng = np.random.default_rng(20260612)
    mask = _full_legal()
    parsed = _parsed_deep_facing()
    tau_max = 0.10
    checked = 0
    for _ in range(200):
        raw = rng.random(N_ACT).astype(np.float32) ** 4   # spiky tails
        p = raw / raw.sum()
        if float(p.max()) <= tau_max:
            continue
        out = apply_commitment_tail_floor(p, mask, parsed, state=None,
                                          tau_max=tau_max)
        assert int(np.argmax(out * mask)) == int(np.argmax(p * mask))
        checked += 1
    assert checked > 100   # the property was actually exercised


# --------------------------------------------------------------------------
# make_live_policy_filter wiring
# --------------------------------------------------------------------------

def test_composed_off_path_never_calls_tail_floor(monkeypatch):
    """tail_floor_tau=None must not even invoke the function — structural
    byte-identity of the OFF path (TG1)."""
    def _boom(*a, **k):
        raise AssertionError("tail floor called on the OFF path")
    monkeypatch.setattr(live_loop, "apply_commitment_tail_floor", _boom)

    # A spot where the tail floor WOULD fire if armed (7.5% ALLIN), but
    # no other floor fires (deep stack, junk hand, facing action).
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.075,
                      A_BET_100: 0.125})
    parsed = _parsed_deep_facing()
    f = make_live_policy_filter(short_stack_threshold_bb=6.0)
    out = f(p, _full_legal(), parsed, state=None)
    assert out is p   # nothing fires → identity through the whole chain


def test_composed_armed_fires_and_logs(capsys):
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.075,
                      A_BET_100: 0.125})
    parsed = _parsed_deep_facing()
    f = make_live_policy_filter(short_stack_threshold_bb=6.0,
                                tail_floor_tau=0.10)
    out = f(p, _full_legal(), parsed, state=None)
    assert out is not p
    assert out[A_ALLIN] == 0.0
    assert abs(out.sum() - 1.0) < 1e-5
    cap = capsys.readouterr().out
    assert "fired=[tail]" in cap
    assert "tail_pruned=[ALLIN:0.075" in cap


def test_composed_armed_fires_last_after_short_stack(monkeypatch):
    """At 4 BB facing action: short-stack floor fires first (masks
    intermediate sizes, renormalizes), THEN the tail floor sees the
    short-stack output and prunes the renormalized low-mass ALLIN."""
    seen = {}
    real = live_loop.apply_commitment_tail_floor

    def _recording(policy, legal_mask, parsed, state, tau_max,
                   discrete_to_chip=None):
        seen["policy_in"] = policy
        return real(policy, legal_mask, parsed, state, tau_max,
                    discrete_to_chip=discrete_to_chip)
    monkeypatch.setattr(live_loop, "apply_commitment_tail_floor", _recording)

    # 4 BB at BB=50, facing 75. fold=0.40 call=0.50 allin=0.037 bets=0.063.
    # Short-stack keeps {FOLD,CALL,ALLIN}/0.937 → ALLIN ≈ 0.0395 < 0.10.
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.50, A_ALLIN: 0.037})
    parsed = {"current_player": 0,
              "money": [200, 1500, 1500, 1500, 1500, 1500],
              "contribution": [25, 50, 100, 0, 0, 0],
              "big_blind": 50,
              "public_cards": "", "private_cards": "7d2c", "street_idx": 0}
    f = make_live_policy_filter(short_stack_threshold_bb=6.0,
                                tail_floor_tau=0.10)
    out = f(p, _full_legal(), parsed, state=None)

    # The tail floor received the POST-short-stack policy (intermediates
    # already zeroed, distribution renormalized) — i.e. it ran last.
    pol_in = seen["policy_in"]
    assert pol_in is not p
    for b in BET_SLOTS:
        assert pol_in[b] == 0.0
    assert abs(float(pol_in.sum()) - 1.0) < 1e-5
    # And it pruned the renormalized ALLIN tail.
    assert out[A_ALLIN] == 0.0
    assert out[A_FOLD] > 0.0
    assert out[A_CALL] > 0.0
    assert abs(out.sum() - 1.0) < 1e-5


# --------------------------------------------------------------------------
# Exact chip-map commitment (adversarial-review fix, EXP_H1)
# --------------------------------------------------------------------------
#
# The fallback formula approximates pot = sum(contribution), which can
# understate universal_poker's real pot (e.g. antes folded into the parsed
# pot but not the contribution vector) and therefore UNDERESTIMATE bet
# commitment. With discrete_to_chip supplied, bet commitment is exact:
# chip ints are whole-hand totals, added = chip - contribution[cp].

def _parsed_pot_understated():
    """Hero stack 1000, contributions sum 70 but the table's real pot is
    150 (rake of the corpus mismatch class: seq-14-style ante absorption).
    to_call = 25."""
    return {"current_player": 0,
            "money": [1000, 1500, 1500, 1500, 1500, 1500],
            "contribution": [25, 50, -5, 0, 0, 0],  # sums to 70
            "big_blind": 50,
            "public_cards": "", "private_cards": "9h2d", "street_idx": 0}


def test_d2c_exact_commit_prunes_dangerous_direction_bet():
    """A bet whose REAL cost is far above the fallback estimate must be
    pruned under exact semantics at a tau where the fallback would keep it.

    Fallback commit(BET_200) = to_call + 2*(pot+to_call) = 25+2*95 = 215
    -> tau = 0.10 * 0.215 = 0.0215. Exact chip map says the bet-to total
    is 425 chips => added = 400 -> tau = 0.10 * 0.400 = 0.040.
    A 3% mass sits between the two thresholds."""
    p = _mk_policy({A_BET_200: 0.03, A_FOLD: 0.50, A_CALL: 0.47})
    parsed = _parsed_pot_understated()
    d2c = {DiscreteAction.FOLD: 0, DiscreteAction.CALL: 1,
           DiscreteAction.BET_200: 425, DiscreteAction.ALLIN: 1025}
    # fallback: NOT pruned (0.03 > 0.0215)
    out_fb = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                         state=None, tau_max=0.10)
    assert out_fb is p
    # exact: pruned (0.03 < 0.040)
    out_ex = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                         state=None, tau_max=0.10,
                                         discrete_to_chip=d2c)
    assert out_ex is not p
    assert float(out_ex[A_BET_200]) == 0.0
    assert abs(float(out_ex.sum()) - 1.0) < 1e-5


def test_d2c_fold_call_chip_ints_ignored_for_bets():
    """chip ints 0 (fold) and 1 (call) are sentinels, never bet totals;
    CALL commitment must come from to_call, not the chip map."""
    p = _mk_policy({A_CALL: 0.02, A_FOLD: 0.50, A_ALLIN: 0.48})
    parsed = _parsed_pot_understated()   # to_call=25, stack=1000
    d2c = {DiscreteAction.FOLD: 0, DiscreteAction.CALL: 1,
           DiscreteAction.ALLIN: 1025}
    # CALL commit_frac = 25/1000 = 0.025 -> tau = 0.0025 < 0.02: kept.
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.10,
                                      discrete_to_chip=d2c)
    assert float(out[A_CALL]) > 0.0


def test_d2c_int_keys_accepted():
    """The captured/replayed chip maps may carry int keys instead of
    DiscreteAction enum keys — both must resolve."""
    p = _mk_policy({A_BET_200: 0.03, A_FOLD: 0.50, A_CALL: 0.47})
    parsed = _parsed_pot_understated()
    d2c = {A_FOLD: 0, A_CALL: 1, A_BET_200: 425, A_ALLIN: 1025}
    out = apply_commitment_tail_floor(p, _full_legal(), parsed,
                                      state=None, tau_max=0.10,
                                      discrete_to_chip=d2c)
    assert float(out[A_BET_200]) == 0.0


def test_composed_filter_advertises_and_threads_d2c(monkeypatch):
    """make_live_policy_filter exposes accepts_d2c=True and forwards the
    map to the tail floor."""
    seen = {}
    real = live_loop.apply_commitment_tail_floor

    def spy(policy, legal_mask, parsed, state, tau_max,
            discrete_to_chip=None):
        seen["d2c"] = discrete_to_chip
        return real(policy, legal_mask, parsed, state, tau_max=tau_max,
                    discrete_to_chip=discrete_to_chip)

    monkeypatch.setattr(live_loop, "apply_commitment_tail_floor", spy)
    f = make_live_policy_filter(tail_floor_tau=0.10)
    assert getattr(f, "accepts_d2c", False) is True
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.20})
    d2c = {A_FOLD: 0, A_CALL: 1, A_ALLIN: 1025}
    f(p, _full_legal(), _parsed_deep_facing(), None, discrete_to_chip=d2c)
    assert seen["d2c"] is d2c


def test_composed_filter_legacy_four_arg_call_still_works():
    """Callers unaware of the d2c protocol use the 4-arg form."""
    f = make_live_policy_filter(tail_floor_tau=0.10)
    p = _mk_policy({A_FOLD: 0.40, A_CALL: 0.40, A_ALLIN: 0.20})
    out = f(p, _full_legal(), _parsed_deep_facing(), None)
    assert abs(float(np.asarray(out).sum()) - 1.0) < 1e-5
