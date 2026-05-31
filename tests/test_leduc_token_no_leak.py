"""§8 information-leakage invariant tests for the Leduc adaptive policy.

These tests are the load-bearing correctness surface of the whole proof:
they verify by construction that the tokenizer cannot leak any bit of any
opponent's private (hole) card into model inputs at any prediction position.

Per arch doc §8 + the Step-3 build order, HARD STOP #1 — these MUST be GREEN
before any training step runs.

Test A — structural:
  * the 17-dim token layout has a hero_card slot but no opp-private-card slot
  * TOKEN_MANIFEST_SHA is frozen (silent reorder => loud test failure)
  * empirically: hero_card slot is exactly zero at every opp token

Test B — counterfactual-equivalence (the proof):
  for >= 1000 sampled (transcript, opp-card-substitution) pairs, replace the
  opp's hole card with EVERY other legal card and assert that the emitted
  token tensor, opp_stats vector, legal_masks, and actor_is_hero array are
  BIT-IDENTICAL between base and substitute. Also verified at the query-
  token level (tokenize_decision mid-hand).

Test C — opp summary stats carry no card information:
  the 6-dim running-stats vector is a pure function of (action, facing_bet)
  per opp decision; the counter object holds only integer counts.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pyspiel

from src.leduc.adaptive import tokens as T

GAME = pyspiel.load_game("leduc_poker")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _random_hand(rng) -> list[int]:
    """Roll a full Leduc hand under uniform-random legal play (incl. chance)."""
    state = GAME.new_initial_state()
    history: list[int] = []
    while not state.is_terminal():
        legal = state.legal_actions()
        a = int(legal[rng.integers(len(legal))])
        history.append(a)
        state.apply_action(a)
    return history


def _deals(history):
    """Return (p0_card, p1_card, public_or_None) by replaying chance nodes."""
    state = GAME.new_initial_state()
    p0 = p1 = pub = None
    dc = 0
    for a in history:
        if state.is_chance_node():
            if dc == 0:
                p0 = a
            elif dc == 1:
                p1 = a
            elif dc == 2:
                pub = a
            dc += 1
        state.apply_action(a)
    return p0, p1, pub


def _substitute_opp_card(history, hero_seat, alt):
    """Replace the opp deal (chance index = 1-hero_seat) with `alt`."""
    new = list(history)
    new[1 - hero_seat] = alt
    return new


# --------------------------------------------------------------------------
# Test A — structural
# --------------------------------------------------------------------------

def test_A_no_opp_private_card_field_in_layout():
    names = {f[0] for f in T.FIELDS}
    assert "hero_card" in names
    forbidden_substrings = ("opp_card", "opp_hole", "opp_private",
                            "villain_card", "opp_rank", "opp_hand")
    for fname, _, _ in T.FIELDS:
        low = fname.lower()
        for bad in forbidden_substrings:
            assert bad not in low, (
                f"§8 violation: token field '{fname}' looks like an opp-private "
                f"slot ('{bad}')"
            )
    # Pin the manifest so any silent reorder/rename fails this test loudly.
    expected_sha = hashlib.sha256(T.TOKEN_MANIFEST.encode()).hexdigest()
    assert T.TOKEN_MANIFEST_SHA == expected_sha
    assert T.TOKEN_MANIFEST.startswith("leduc-adaptive-token-v1|")
    # The 9 documented fields cover the full F_RAW=17 dim layout.
    total = sum(w for _, _, w in T.FIELDS)
    assert total == T.F_RAW


def test_A_stats_manifest_v3_pinned():
    """v3 opp_stats layout (20 dims, per-street split): no opp-card-implicating
    field, manifest SHA-256 pinned, dim count matches, preflop+postflop
    blocks both present with matching field semantics."""
    assert T.F_STATS == 20
    forbidden_substrings = ("card", "hole", "rank", "suit", "private", "deal")
    for fname, _ in T.STATS_FIELDS:
        low = fname.lower()
        for bad in forbidden_substrings:
            assert bad not in low, (
                f"§8 violation: stats field '{fname}' looks like a card slot "
                f"('{bad}')"
            )
    expected_sha = hashlib.sha256(T.STATS_MANIFEST.encode()).hexdigest()
    assert T.STATS_MANIFEST_SHA == expected_sha
    assert T.STATS_MANIFEST.startswith("leduc-adaptive-opp-stats-v3|")
    # 20 field indices unique and consecutive 0..19.
    indices = sorted(idx for _, idx in T.STATS_FIELDS)
    assert indices == list(range(20))
    # Preflop block at 0..7, postflop block at 8..15.
    names = {f[0]: f[1] for f in T.STATS_FIELDS}
    assert all(names[f"preflop_{stem}"] < 8
               for stem in ("facing_fold_rate_smoothed",
                            "facing_call_rate_smoothed",
                            "facing_raise_rate_smoothed",
                            "aggression_factor_smoothed",
                            "open_raise_rate_smoothed",
                            "confidence_weight",
                            "raw_raise_fraction",
                            "raw_fold_fraction"))
    assert all(8 <= names[f"postflop_{stem}"] < 16
               for stem in ("facing_fold_rate_smoothed",
                            "facing_call_rate_smoothed",
                            "facing_raise_rate_smoothed",
                            "aggression_factor_smoothed",
                            "open_raise_rate_smoothed",
                            "confidence_weight",
                            "raw_raise_fraction",
                            "raw_fold_fraction"))
    # Last-action snapshot at 16..19.
    for fname in ("last_action_is_fold", "last_action_is_call",
                  "last_action_is_raise", "no_actions_yet"):
        assert 16 <= names[fname] < 20


def test_A_hero_card_slot_zero_at_every_opp_token():
    """Empirically: the hero_card slot tok[9:12] is exactly 0 at opp tokens.
    This proves the slot's semantic identity is fixed to HERO — it isn't a
    'current actor's card' slot that would silently mirror opp info."""
    rng = np.random.default_rng(20260531)
    n_opp_tokens = 0
    for _ in range(400):
        h = _random_hand(rng)
        for hero_seat in (0, 1):
            out = T.tokenize_full_hand(h, hero_seat)
            toks = out["tokens"]
            is_hero = out["actor_is_hero"]
            opp_rows = toks[~is_hero]
            if opp_rows.size:
                assert np.all(opp_rows[:, 9:12] == 0.0), (
                    "§8 violation: hero_card slot non-zero on an opp token"
                )
                n_opp_tokens += int((~is_hero).sum())
    assert n_opp_tokens > 0


# --------------------------------------------------------------------------
# Test B — counterfactual equivalence (load-bearing)
# --------------------------------------------------------------------------

def test_B_counterfactual_opp_card_substitution_bit_identical():
    """For >=1000 (transcript, substitution) pairs: substituting the opp's
    hole card for any other legal card leaves the full tokenizer output
    BIT-IDENTICAL. This is the constructive proof of §8."""
    rng = np.random.default_rng(424242)
    pairs = 0
    mismatches: list[str] = []
    for hand_idx in range(500):
        base_history = _random_hand(rng)
        p0, p1, pub = _deals(base_history)
        for hero_seat in (0, 1):
            opp_card = p1 if hero_seat == 0 else p0
            hero_card = p0 if hero_seat == 0 else p1
            base_out = T.tokenize_full_hand(base_history, hero_seat)
            forbidden = {hero_card, opp_card}
            if pub is not None:
                forbidden.add(pub)
            for alt in range(6):
                if alt in forbidden:
                    continue
                alt_history = _substitute_opp_card(base_history, hero_seat, alt)
                alt_out = T.tokenize_full_hand(alt_history, hero_seat)
                ok = (
                    np.array_equal(base_out["tokens"], alt_out["tokens"])
                    and np.array_equal(base_out["opp_stats"], alt_out["opp_stats"])
                    and np.array_equal(base_out["legal_masks"], alt_out["legal_masks"])
                    and np.array_equal(base_out["actor_is_hero"], alt_out["actor_is_hero"])
                    and base_out["n_tokens"] == alt_out["n_tokens"]
                )
                pairs += 1
                if not ok:
                    diff = np.where(base_out["tokens"] != alt_out["tokens"])
                    mismatches.append(
                        f"hand_idx={hand_idx} hero={hero_seat} "
                        f"opp_orig={opp_card} alt={alt} diff@{diff}"
                    )
    assert pairs >= 1000, f"need >=1000 pairs, got {pairs}"
    assert not mismatches, (
        f"§8 COUNTERFACTUAL LEAK: {len(mismatches)} of {pairs} pairs differ. "
        f"first 3 = {mismatches[:3]}"
    )
    # Surface the pair count for the test report.
    print(f"\n[Test B] verified {pairs} counterfactual pairs bit-identical")


def test_B_counterfactual_query_token_invariance():
    """Same invariance at the QUERY-token level (tokenize_decision mid-hand).
    Replays each base hand step-by-step; at each hero decision node, builds
    a query under both opp-card values and asserts bit-identical output."""
    rng = np.random.default_rng(1337)
    query_pairs = 0
    mismatches: list[str] = []
    for hand_idx in range(200):
        base_history = _random_hand(rng)
        p0, p1, pub = _deals(base_history)
        for hero_seat in (0, 1):
            opp_card = p1 if hero_seat == 0 else p0
            hero_card = p0 if hero_seat == 0 else p1
            forbidden = {hero_card, opp_card}
            if pub is not None:
                forbidden.add(pub)
            alts = [c for c in range(6) if c not in forbidden]
            if not alts:
                continue
            alt = alts[0]  # one substitution per (hand, seat) is enough here
            alt_history = _substitute_opp_card(base_history, hero_seat, alt)

            # Walk both states in lockstep; at every hero decision node, query.
            base_st = GAME.new_initial_state()
            alt_st = GAME.new_initial_state()
            for a_base, a_alt in zip(base_history, alt_history):
                if (not base_st.is_chance_node()
                        and not base_st.is_terminal()
                        and base_st.current_player() == hero_seat):
                    qb = T.tokenize_decision(base_st)
                    qa = T.tokenize_decision(alt_st)
                    ok = (
                        np.array_equal(qb["tokens"], qa["tokens"])
                        and qb["query_idx"] == qa["query_idx"]
                        and np.array_equal(qb["legal_mask"], qa["legal_mask"])
                        and np.array_equal(qb["opp_stats"], qa["opp_stats"])
                        and qb["legal_actions"] == qa["legal_actions"]
                    )
                    query_pairs += 1
                    if not ok:
                        diff = np.where(qb["tokens"] != qa["tokens"])
                        mismatches.append(
                            f"hand_idx={hand_idx} hero={hero_seat} "
                            f"opp_orig={opp_card} alt={alt} diff@{diff}"
                        )
                base_st.apply_action(a_base)
                alt_st.apply_action(a_alt)
    assert not mismatches, (
        f"§8 QUERY-TOKEN LEAK: {len(mismatches)} of {query_pairs} pairs differ. "
        f"first 3 = {mismatches[:3]}"
    )
    print(f"\n[Test B-query] verified {query_pairs} query-token pairs")


# --------------------------------------------------------------------------
# Test C — opp summary stats carry no card information
# --------------------------------------------------------------------------

def test_C_opp_stats_is_pure_function_of_actions_facing_and_street():
    """The 20-dim v3 opp_stats vector is derived entirely from
    (action, facing_bet, street, previous_action) per opp decision. The
    counter object holds only integer counts and the last action id; cards
    never enter. Per-street blocks are routed by the `street` argument.

    Stream chosen to exercise both blocks and the spot-check arithmetic:
        preflop:  RAISE (open), CALL (facing)
        postflop: FOLD (facing)
    Preflop counts: notfacing_raise=1, facing_call=1, p_n=2
    Postflop counts: facing_fold=1, f_n=1
    Last action: FOLD (postflop, facing).
    """
    stream = [
        (T.RAISE, False, 1),
        (T.CALL, True, 1),
        (T.FOLD, True, 2),
    ]
    a = T._OppCounters()
    b = T._OppCounters()
    for act, fb, street in stream:
        a.update(act, fb, street)
        b.update(act, fb, street)
    va = a.vector()
    vb = b.vector()
    assert np.array_equal(va, vb)
    assert va.shape == (T.F_STATS,) == (20,)

    # Spot check preflop block (dims 0-7):
    #   facing_fold=0, facing_call=1, facing_raise=0 -> f=1
    #   notfacing_call=0, notfacing_raise=1 -> nf=1
    #   p_n=2, all_raise=1, all_call=1
    assert abs(va[0] - 1.0 / 4.0) < 1e-6  # p_facing_fold_rate = 1/(1+3)
    assert abs(va[1] - 2.0 / 4.0) < 1e-6  # p_facing_call_rate = 2/4
    assert abs(va[2] - 1.0 / 4.0) < 1e-6  # p_facing_raise_rate
    assert abs(va[3] - 2.0 / 4.0) < 1e-6  # p_aggression_factor = (1+1)/(1+1+2)
    assert abs(va[4] - 2.0 / 3.0) < 1e-6  # p_open_raise = (1+1)/(1+2)
    assert abs(va[6] - 1.0 / 2.0) < 1e-6  # p_raw_raise_fraction = 1/2
    assert abs(va[7] - 0.0) < 1e-6        # p_raw_fold_fraction = 0/1

    # Spot check postflop block (dims 8-15):
    #   facing_fold=1, facing_call=0, facing_raise=0 -> f=1
    #   notfacing_call=0, notfacing_raise=0 -> nf=0
    #   f_n=1, all_raise=0, all_call=0
    assert abs(va[8]  - 2.0 / 4.0) < 1e-6  # f_facing_fold_rate = 2/4
    assert abs(va[9]  - 1.0 / 4.0) < 1e-6  # f_facing_call_rate = 1/4
    assert abs(va[10] - 1.0 / 4.0) < 1e-6  # f_facing_raise_rate = 1/4
    assert abs(va[11] - 1.0 / 2.0) < 1e-6  # f_aggression_factor = 1/2
    assert abs(va[12] - 1.0 / 2.0) < 1e-6  # f_open_raise (denom n+2 = 2) = 1/2
    assert abs(va[14] - 0.0) < 1e-6        # f_raw_raise_fraction = 0/1
    assert abs(va[15] - 1.0) < 1e-6        # f_raw_fold_fraction = 1/1

    # Last-action snapshot: last action was FOLD.
    assert va[16] == 1.0 and va[17] == 0.0 and va[18] == 0.0
    assert va[19] == 0.0  # not-no-actions-anymore

    # No-info baseline = v3 prior.
    empty = T._OppCounters().vector()
    assert np.array_equal(T.empty_opp_stats(), empty)
    assert empty[19] == 1.0  # no_actions_yet (across BOTH streets)
    assert empty[16] == 0.0 and empty[17] == 0.0 and empty[18] == 0.0

    # Routing check: same action+facing but different street routes to
    # different blocks.
    p_only = T._OppCounters()
    p_only.update(T.RAISE, False, 1)
    f_only = T._OppCounters()
    f_only.update(T.RAISE, False, 2)
    vp = p_only.vector()
    vf = f_only.vector()
    # The preflop-routed update should not change postflop dims and vice versa.
    assert vp[4] != empty[4]    # preflop open_raise rate changed
    assert vp[12] == empty[12]  # postflop open_raise rate unchanged
    assert vf[12] != empty[12]
    assert vf[4] == empty[4]

    # All counter slots are plain integers — no card index can hide there.
    for k, v in vars(a).items():
        assert isinstance(v, int), f"non-integer slot {k}={v!r} in _OppCounters"

    # Source inspection: no card-related symbols inside _OppCounters.
    import inspect
    src = inspect.getsource(T._OppCounters)
    assert "card_rank" not in src
    assert "hero_card" not in src
    assert "public" not in src
    assert "hole" not in src
    assert "deal" not in src
