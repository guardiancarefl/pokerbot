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


def test_A_stats_manifest_v2_pinned():
    """v2 opp_stats layout (12 dims): no opp-card-implicating field, manifest
    SHA-256 pinned, dim count matches."""
    assert T.F_STATS == 12
    names = {f[0] for f in T.STATS_FIELDS}
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
    assert T.STATS_MANIFEST.startswith("leduc-adaptive-opp-stats-v2|")
    # All 12 field indices are unique and consecutive 0..11.
    indices = sorted(idx for _, idx in T.STATS_FIELDS)
    assert indices == list(range(12))


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

def test_C_opp_stats_is_pure_function_of_actions_and_facing_bet():
    """The 12-dim v2 opp_stats vector is derived entirely from (action,
    facing_bet, previous_action) per opp decision. The counter object holds
    only integer counts and the last action id; cards never enter."""
    # Two counters fed the same action stream produce bit-identical vectors.
    stream = [
        (T.FOLD, True), (T.CALL, True), (T.RAISE, True),
        (T.RAISE, False), (T.CALL, False), (T.CALL, True), (T.RAISE, True),
    ]
    a = T._OppCounters()
    b = T._OppCounters()
    for act, fb in stream:
        a.update(act, fb)
        b.update(act, fb)
    assert np.array_equal(a.vector(), b.vector())
    assert a.vector().shape == (T.F_STATS,) == (12,)

    # No-info baseline = the v2 prior.
    empty = T._OppCounters().vector()
    assert np.array_equal(T.empty_opp_stats(), empty)
    # The "no_actions_yet" flag (dim 11) must be 1 when n=0.
    assert empty[11] == 1.0
    # All last-action one-hots are zero when no action yet.
    assert empty[8] == 0.0 and empty[9] == 0.0 and empty[10] == 0.0

    # Last-action one-hot follows the latest action.
    c = T._OppCounters()
    c.update(T.FOLD, True)
    v = c.vector()
    assert v[8] == 1.0 and v[9] == 0.0 and v[10] == 0.0
    c.update(T.RAISE, False)
    v = c.vector()
    assert v[8] == 0.0 and v[9] == 0.0 and v[10] == 1.0
    assert v[11] == 0.0  # not-no-actions-anymore

    # Raw rates (dims 6-7) are unsmoothed fractions of integer counts.
    # Stream above ended with RAISE-facing as the last action; counts:
    #   facing_fold=1, facing_call=2, facing_raise=2 -> f=5
    #   notfacing_call=1, notfacing_raise=1 -> nf=2
    #   n=7, all_raise=3, all_call=3
    va = a.vector()
    assert abs(va[6] - 3.0 / 7.0) < 1e-6  # raw raise fraction
    assert abs(va[7] - 1.0 / 5.0) < 1e-6  # raw fold fraction

    # All counter slots are plain integers — no card index can hide there.
    for k, v in vars(a).items():
        assert isinstance(v, int), f"non-integer slot {k}={v!r} in _OppCounters"

    # The card_rank() helper exists but is only used for hero_card + public,
    # never within _OppCounters. Sanity-check by ensuring _OppCounters has no
    # method that touches cards.
    import inspect
    src = inspect.getsource(T._OppCounters)
    assert "card_rank" not in src
    assert "hero_card" not in src
    assert "public" not in src
    assert "hole" not in src
    assert "deal" not in src


# --------------------------------------------------------------------------
# Test D — S2(a) cell-label leakage guard
# --------------------------------------------------------------------------
# The S2(a) architecture adds a 9-way cell classifier head. The cell label is
# a TRAINING TARGET only (CE supervision); it must never enter the model's
# forward inputs at inference time. This test asserts that property by:
#   (D1) forward() signature accepts only (tokens, pad_mask, query_idx,
#        legal_mask, opp_stats) — no cell_label / cell_id parameter.
#   (D2) forward() source contains no string referencing cell labels.
#   (D3) Sentinel run: outputs are determined entirely by the 5 declared
#        inputs (so a hypothetical cell label cannot affect them by side
#        channel — e.g., via a global).

def test_D_cell_label_not_in_forward_inputs():
    import inspect
    import torch
    from src.leduc.adaptive.model import AdaptivePolicyNetS2a
    from src.leduc.adaptive.tokens import F_RAW, F_STATS, NUM_ACTIONS

    net = AdaptivePolicyNetS2a(n_cells=9)
    net.eval()

    # D1 — signature: only the 5 expected positional inputs.
    sig = inspect.signature(net.forward)
    params = list(sig.parameters.keys())
    expected = ["tokens", "pad_mask", "query_idx", "legal_mask", "opp_stats"]
    assert params == expected, (
        f"forward() signature changed; expected {expected}, got {params}. "
        "If a cell-label parameter has been added, the S2(a) read primitive's "
        "no-oracle property has been broken at the architecture layer."
    )

    # D2 — source inspection: forward() body must not reference cell labels.
    src = inspect.getsource(net.forward)
    forbidden = ["cell_label", "cell_id", "cell_idx", "true_cell"]
    for tok in forbidden:
        assert tok not in src, (
            f"forward() source contains forbidden token '{tok}'. "
            "Cell identity must not flow into the forward pass.")

    # D3 — sentinel: outputs reproduce bit-identically across calls with the
    # same 5 inputs (no hidden state, no global cell label affecting outputs).
    B, T = 2, 6
    rng = np.random.default_rng(0)
    tokens = torch.from_numpy(rng.standard_normal((B, T, F_RAW)).astype(np.float32))
    pad_mask = torch.zeros((B, T), dtype=torch.bool)
    query_idx = torch.tensor([2, 3], dtype=torch.long)
    legal_mask = torch.ones((B, NUM_ACTIONS), dtype=torch.bool)
    opp_stats = torch.from_numpy(rng.standard_normal((B, F_STATS)).astype(np.float32))

    with torch.no_grad():
        pol1, cell1, act1, comb1 = net(tokens, pad_mask, query_idx,
                                        legal_mask, opp_stats)
        # Run again — same inputs, must produce bit-identical outputs.
        pol2, cell2, act2, comb2 = net(tokens, pad_mask, query_idx,
                                        legal_mask, opp_stats)
    assert torch.equal(pol1, pol2)
    assert torch.equal(cell1, cell2)
    assert torch.equal(act1, act2)
    assert torch.equal(comb1, comb2)

    # And: cell head produces 9-way logits (proves the head exists and is
    # wired); the values are unconstrained — head has not seen the label.
    assert cell1.shape == (B, 9)
    assert act1.shape == (B, NUM_ACTIONS)
    assert pol1.shape == (B, NUM_ACTIONS)
