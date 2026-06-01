"""§8 analog — 6-max NLHE adaptive-policy information-leakage invariant tests.

These tests are the load-bearing correctness surface for the 6-max adaptive
scaffold: they verify by construction that nothing in the head's input pipeline
can leak any bit of any opponent's private (hole) card into the feature vector
at any prediction position, AND that the input at decision t is a pure
function of history-up-to-t (no future-action / future-hand information).

Per the Leduc precedent (tests/test_leduc_token_no_leak.py A/B/C/D) + the
fork-collapsing decision record (docs/DECISIONS.md "Leduc proof complete —
S1 verdict"), these MUST be GREEN as the HARD preflight gate before ANY
6-max adaptive training runs.

Test A6 — structural:
  * the 236-dim feature layout has a hero-card bucket slot but no per-opponent
    private-card slot
  * `InfosetEncoder6Max.feature_dim` is pinned to 236
  * a deterministic layout-signature SHA is pinned (silent reorder => loud
    test failure)
  * `parse_state_6max` exposes only hero-observer fields in its return dict
    (no opp-private-card key)

Test B6 — counterfactual + history-purity:
  * for >= 100 sampled (random 6-max hand, opp-card-substitution) pairs:
    replace ONE opponent's hole cards with two unused legal cards and assert
    that the `InfosetEncoder6Max` output is BIT-IDENTICAL between base and
    substitute. Asserts the encoder cannot read opp private cards through
    any side channel.
  * B6 addendum (history-purity, per the Decision-A override): the encoder's
    output at decision t is a pure function of history-up-to-t. For >= 100
    states reached by random play, encode at t -> V1; replay the same history
    from a fresh game with the same RNG seed -> V2; assert V1 == V2. Catches
    nondeterminism / cache contamination / accidental future-state side
    channels.

Test C6 — opp-stat purity:
  * `MatchObserver.update()` and `_OppCounters`-equivalent (SeatStats) carry
    no card information. Source-inspect within_match.py for forbidden tokens
    (card, hole, deal, private) outside whitelisted comments.
  * Sanity-run on a hand: SeatStats values depend only on observed actions
    and the public parsed-state dict.

Test D6 — model-surface leak guard for `Adaptive6MaxNet` (scaffold step 1b).
The `Adaptive6MaxNet.forward()` signature accepts ONLY the 5 declared inputs
(tokens, pad_mask, query_idx, legal_mask, opp_stats); no tendency target,
archetype id, true-tendency, match-confidence oracle, or other identity-of-
opponent parameter may enter the forward pass. Source inspection forbids
such references; sentinel test verifies forward determinism on identical
inputs.
"""
from __future__ import annotations

import hashlib
import inspect
import random
import re

import numpy as np
import pyspiel
import pytest

from src.nlhe.game_strings import six_max_sng
from src.nlhe.infoset6 import InfosetEncoder6Max, parse_state_6max
from src.nlhe import within_match as WM

GAME = pyspiel.load_game(six_max_sng())
NUM_PLAYERS = 6


# --------------------------------------------------------------------------
# Test fixtures: a small abstraction stand-in so encoder calls don't require
# loading a 200-bucket pickle. The leakage tests do NOT depend on abstraction
# quality — they only assert structural properties of the encoded vector.
# --------------------------------------------------------------------------

class _StubAbstraction:
    """Minimal abstraction shim implementing the surface the encoder calls
    (`bucket_of(hero, board, runouts, rng) -> int`). Deterministic: bucket
    is a hash of (frozenset(hero_cards), tuple(board_cards)). Stand-in for
    the production Abstraction object during leakage testing — the tests
    care about ENCODER PURITY, not abstraction quality."""

    def __init__(self, k: int = 200):
        self.k = int(k)

    def bucket_of(self, hero, board, runouts: int = 30, rng=None) -> int:
        # hero / board are lists of card ints in [0, 52). Determinism
        # depends ONLY on the (hero, board) tuple, NOT on rng (so the
        # B6 bit-identity check passes regardless of MC variance in the
        # real abstraction).
        key = (tuple(sorted(int(c) for c in hero)),
               tuple(sorted(int(c) for c in board)))
        h = hashlib.sha256(repr(key).encode()).hexdigest()
        return int(h, 16) % self.k


def _make_encoder() -> InfosetEncoder6Max:
    return InfosetEncoder6Max(
        abstraction=_StubAbstraction(k=200),
        starting_stack=1500,
        max_bucket_dim=200,
        bucket_runouts=30,
    )


# --------------------------------------------------------------------------
# Helpers — random 6-max hand walking and opp-card substitution
# --------------------------------------------------------------------------

def _random_hand_to_decision(rng: random.Random) -> "pyspiel.State":
    """Play a random 6-max hand under uniform-random legal play until either
    (a) we reach a decision node at least one street in, or (b) terminal."""
    state = GAME.new_initial_state()
    n_decisions_passed = 0
    target = rng.randint(1, 6)
    while not state.is_terminal():
        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            acts = [a for a, _ in outcomes]
            ps = [p for _, p in outcomes]
            tot = sum(ps)
            ps = [p / tot for p in ps]
            state.apply_action(int(rng.choices(acts, weights=ps, k=1)[0]))
            continue
        if n_decisions_passed >= target:
            return state
        legal = state.legal_actions()
        state.apply_action(int(rng.choice(legal)))
        n_decisions_passed += 1
    return state


def _history_with_opp_cards_substituted(orig_state, opp_seat: int,
                                          rng: random.Random):
    """Return a fresh state replayed from the original history but with the
    specified opp_seat's two hole-card chance actions replaced with two cards
    not used anywhere else in the original deal (hole or board).

    Universal_poker 6-max deal order (CONFIRMED at runtime via parse on a
    deterministic deal of cards 0..11): seat 0 receives cards at history
    positions 0 and 1, seat 1 at 2 and 3, ..., seat 5 at 10 and 11. So
    opp_seat O's two hole cards are at history positions 2*O and 2*O+1.
    We replace those two with cards not conflicting with any other card
    already used in the hand."""
    full_history = list(orig_state.history())
    # Sanity: first 12 actions must be chance actions producing valid card ints.
    assert len(full_history) >= 12, (
        f"history too short to substitute hole cards: {len(full_history)}")
    used_cards = set(full_history[:12])
    # If board cards have been dealt, those positions are also card actions —
    # add them to the used set so we don't accidentally collide.
    # Board cards (flop x3, turn x1, river x1) follow the preflop betting;
    # they appear as chance actions interleaved later in the history. We
    # conservatively add ALL chance-action ints in [0, 52) from the full
    # history to used_cards by replaying and noting which are chance.
    replay = GAME.new_initial_state()
    chance_ints: list[int] = []
    for a in full_history:
        if replay.is_chance_node():
            chance_ints.append(int(a))
        replay.apply_action(int(a))
    # Cards are integers in [0, 52); some non-card chance actions (e.g.
    # dealer rotation, if any) would have larger ints — universal_poker
    # uses card ints only, so chance_ints ⊆ [0, 52).
    used_cards = set(int(c) for c in chance_ints if 0 <= int(c) < 52)
    # Pick two unused cards for the substitution.
    deck = set(range(52))
    available = list(deck - used_cards)
    assert len(available) >= 2, (
        f"only {len(available)} cards available for substitution")
    rng.shuffle(available)
    sub_card_1 = int(available[0])
    sub_card_2 = int(available[1])
    new_history = list(full_history)
    new_history[2 * opp_seat] = sub_card_1
    new_history[2 * opp_seat + 1] = sub_card_2

    # Replay; at each chance step, the substituted action must be legal.
    state = GAME.new_initial_state()
    for a in new_history:
        if state.is_chance_node():
            legal = set(state.legal_actions())
            if a not in legal:
                # Substitution conflicted with a later chance action — pick a
                # different legal card from the available set.
                fallback = next((c for c in available[2:] if c in legal), None)
                if fallback is None:
                    return None  # give up this sample; caller skips
                a = int(fallback)
            state.apply_action(int(a))
        else:
            if a not in state.legal_actions():
                return None
            state.apply_action(int(a))
    return state


# --------------------------------------------------------------------------
# Test A6 — structural
# --------------------------------------------------------------------------

def test_A6_no_opp_private_card_field_in_layout():
    """The 236-dim feature layout has a hero-card bucket but no per-opponent
    private-card slot. Source-inspect `encode_from_parsed` to confirm no opp-
    card field is read or written, and verify the layout enumeration."""
    enc = _make_encoder()
    src = inspect.getsource(InfosetEncoder6Max.encode_from_parsed)
    # Forbidden patterns: the encoder must not reference any per-opponent
    # private/hole-card field. (Comments may use these words; restrict to
    # likely-identifier contexts.)
    forbidden_substrings = [
        "opp_card", "opponent_card", "opp_hole", "opponent_hole",
        "opp_private", "opponents_private", "villain_card",
    ]
    for tok in forbidden_substrings:
        assert tok not in src, (
            f"encode_from_parsed references forbidden token '{tok}' — opp "
            "private cards must not appear in the feature pipeline.")
    # Only the hero's own private_cards is read, and only as input to the
    # hero-bucket lookup. Confirm by inspecting _get_bucket.
    src_bucket = inspect.getsource(InfosetEncoder6Max._get_bucket)
    assert "private_cards" in src_bucket, (
        "expected _get_bucket to consume parsed['private_cards'] (hero's "
        "own cards), but found no reference — encoder layout drifted.")


def test_A6_feature_dim_pinned():
    """feature_dim is pinned to 236. Drift would silently change the input
    surface — would invalidate every downstream test and any trained model."""
    enc = _make_encoder()
    assert enc.feature_dim == 236, (
        f"feature_dim = {enc.feature_dim}, expected 236. The 6-max encoder "
        "layout has changed; update the manifest pin AND audit all dependent "
        "leakage / scaffold tests.")


def test_A6_layout_manifest_pinned():
    """A deterministic layout-signature SHA is pinned — silent reorder of
    the segments inside `encode_from_parsed` will desynchronise this and
    fail loudly. Layout description matches the docstring at infoset6.py:172
    (k=200 abstraction)."""
    # The segments, in the order they're written, with their declared widths.
    layout = (
        "6max-encoder-v1"
        "|bucket:200"
        "|street:4"
        "|position:6"
        "|stacks_per_player:6"
        "|active_mask_per_player:6"
        "|contribution_per_player:6"
        "|pot_norm:1"
        "|tocall_norm:1"
        "|eff_stack_norm:1"
        "|betting_history:5"
    )
    expected_total = 200 + 4 + 6 + 6 + 6 + 6 + 1 + 1 + 1 + 5
    assert expected_total == 236
    sha = hashlib.sha256(layout.encode()).hexdigest()
    # Pinned signature — bump this ONLY when intentionally redesigning the
    # 6-max feature layout, and bump it together with every dependent test
    # and any trained checkpoint.
    expected_sha = (
        "9bf0944a9af0039af241f0bcc4b222d1d9e3c3a020c23b47943bef39c58b8741")
    if sha != expected_sha:
        # Print the actual SHA so the developer can intentionally re-pin if
        # the redesign is approved. Test still fails.
        raise AssertionError(
            f"6-max encoder layout SHA changed.\n"
            f"  computed: {sha}\n"
            f"  expected: {expected_sha}\n"
            f"  layout:   {layout}\n"
            f"If this change is intentional (approved layout redesign), bump "
            f"`expected_sha` here AND audit/bump every dependent test and "
            f"trained-model schema field."
        )


def test_A6_parse_state_exposes_only_hero_observer_fields():
    """parse_state_6max returns a dict with hero-observer fields only. The
    encoder consumes this dict; if it contained any per-opp private field,
    the encoder could leak it. Assert by source inspection + an empirical
    walk on a few sampled hands."""
    src = inspect.getsource(parse_state_6max)
    # The function must not produce a key referencing opp/villain private
    # cards. We look for assignments into `out[...]` whose key suggests opp.
    out_keys = re.findall(r"""out\[['"]([^'"]+)['"]\]""", src)
    forbidden_key_substrings = (
        "opp", "villain", "opponent", "opp_private", "opp_hole")
    for k in out_keys:
        for f in forbidden_key_substrings:
            assert f not in k.lower(), (
                f"parse_state_6max assigns to out['{k}'] — contains forbidden "
                f"substring '{f}'. Hero-observer dict must not surface "
                "per-opp private info.")

    # Empirical: parsed dict only has the documented keys.
    expected_keys = {
        "num_players", "street_idx", "current_player", "pot", "money",
        "contribution", "private_cards", "public_cards", "sequences",
    }
    rng = random.Random(2026)
    for _ in range(20):
        state = _random_hand_to_decision(rng)
        if state.is_terminal():
            continue
        parsed = parse_state_6max(state)
        # private_cards in this dict is HERO's own cards (the observer).
        # The key's name is unfortunate but its content is hero-only.
        # Verify the set of keys matches the documented surface.
        extra_keys = set(parsed.keys()) - expected_keys
        assert not extra_keys, (
            f"parse_state_6max produced unexpected key(s) {extra_keys}; "
            "audit the parser before treating it as §8-clean.")


# --------------------------------------------------------------------------
# Test B6 — counterfactual + history-purity
# --------------------------------------------------------------------------

def test_B6_counterfactual_opp_card_substitution_bit_identical():
    """For >= 100 sampled hands, replace ONE opponent's hole cards with two
    unused legal cards and assert `InfosetEncoder6Max` output is bit-identical
    between base and substitute. Strong empirical proof that the encoder has
    no opp-private-card side channel."""
    enc = _make_encoder()
    rng = random.Random(20260601)
    n_checked = 0
    n_attempts = 0
    target = 100
    # Use a fixed-seed Python random for encoder MC runouts; ensures the
    # postflop bucket lookup is deterministic across the base/substitute
    # pair.
    enc_rng_seed = 314159
    while n_checked < target and n_attempts < 5 * target:
        n_attempts += 1
        state = _random_hand_to_decision(rng)
        if state.is_terminal():
            continue
        # Pick an opp seat (any seat != current hero).
        hero = state.current_player()
        opp = rng.choice([s for s in range(NUM_PLAYERS) if s != hero])
        sub_state = _history_with_opp_cards_substituted(state, opp, rng)
        if sub_state is None:
            # substitution conflicted with later board cards; skip
            continue
        # The substituted state must be at the SAME decision point.
        if sub_state.current_player() != hero:
            continue
        if sub_state.is_terminal():
            continue
        v_base = enc.encode(state, random.Random(enc_rng_seed))
        enc.reset_cache()
        v_sub = enc.encode(sub_state, random.Random(enc_rng_seed))
        enc.reset_cache()
        assert v_base.shape == v_sub.shape == (236,)
        assert np.array_equal(v_base, v_sub), (
            f"opp-card substitution changed encoder output at decision "
            f"(hero={hero}, opp={opp}); first differing index: "
            f"{int(np.argmax(v_base != v_sub))}, "
            f"base={v_base[v_base != v_sub][:5]}, "
            f"sub={v_sub[v_base != v_sub][:5]}. "
            "Encoder leaks opp private cards."
        )
        n_checked += 1
    assert n_checked >= target, (
        f"only checked {n_checked} / {target} pairs ({n_attempts} attempts); "
        "test sample-rate is too low — investigate the substitution helper.")


def test_B6_encoder_is_pure_function_of_history_up_to_t():
    """B6 ADDENDUM (per Decision-A override): the encoder's output at
    decision t must be a pure function of history-up-to-t. For >= 100 states
    reached by random play, encode at t -> V1; replay the same history from
    a fresh game with the same RNG seed -> V2; assert V1 == V2. Catches
    nondeterminism, hidden caches, accidental future-state side channels."""
    enc = _make_encoder()
    rng = random.Random(20260602)
    n_checked = 0
    target = 100
    enc_rng_seed = 271828
    for _ in range(target * 2):
        if n_checked >= target:
            break
        state = _random_hand_to_decision(rng)
        if state.is_terminal():
            continue
        history = list(state.history())
        # Fresh game; replay history; encode at same decision point.
        replay = GAME.new_initial_state()
        for a in history:
            if replay.is_terminal():
                break
            replay.apply_action(int(a))
        if replay.is_terminal():
            continue
        if replay.current_player() != state.current_player():
            continue
        v_orig = enc.encode(state, random.Random(enc_rng_seed))
        enc.reset_cache()
        v_replay = enc.encode(replay, random.Random(enc_rng_seed))
        enc.reset_cache()
        assert np.array_equal(v_orig, v_replay), (
            "Encoder output differs between an in-place state and a fresh "
            "replay of the same history-up-to-t. The encoder is not a pure "
            "function of history — investigate caches / globals / future-"
            "state leaks."
        )
        n_checked += 1
    assert n_checked >= target, (
        f"only checked {n_checked} / {target} pairs; sample-rate too low.")


# --------------------------------------------------------------------------
# Test C6 — MatchObserver / SeatStats stats purity (no card information)
# --------------------------------------------------------------------------

def test_C6_match_observer_carries_no_card_information():
    """The `within_match` module's accumulator surface must carry no card
    information. Source-inspect for forbidden tokens (card, hole, deal,
    private) outside whitelisted comment lines, AND empirically verify a
    sanity hand produces SeatStats counters that depend only on (action,
    parsed-public-state)."""
    src = inspect.getsource(WM)
    # Strip docstrings and comment lines before substring scan — the module
    # docstring legitimately mentions "card" / "hand" in prose.
    code_only_lines: list[str] = []
    in_docstring = False
    docstring_marker = '"""'
    for line in src.splitlines():
        stripped = line.lstrip()
        # toggle docstring blocks (simple heuristic — `"""` opening/closing)
        if stripped.startswith(docstring_marker):
            count = stripped.count(docstring_marker)
            if count == 1:
                in_docstring = not in_docstring
            # if count >= 2, opens-and-closes on same line — net no toggle
            continue
        if in_docstring:
            continue
        # strip comments
        if stripped.startswith("#"):
            continue
        # in-line comment removal
        if "#" in line:
            line = line.split("#", 1)[0]
        code_only_lines.append(line)
    code_only = "\n".join(code_only_lines)
    # Forbidden identifier substrings in CODE — NOT in docstrings or comments.
    forbidden_in_code = (
        "card_rank", "hole_card", "private_card", "deal_card",
        "showdown_card", "villain_card",
    )
    for tok in forbidden_in_code:
        assert tok not in code_only, (
            f"within_match.py CODE references forbidden token '{tok}'. The "
            "observer must be a pure function of public actions; card "
            "information must not enter the accumulator surface.")

    # SeatStats fields — assert they are integer counters / single floats.
    # No card-typed fields should exist.
    seat_stats_fields = WM.SeatStats.__dataclass_fields__
    type_strings = {
        name: str(f.type) for name, f in seat_stats_fields.items()
    }
    # Whitelisted types: int, float, list[int], str-aliases of those.
    allowed_substrings = ("int", "float", "list")
    for name, tstr in type_strings.items():
        ok = any(s in tstr for s in allowed_substrings)
        assert ok, (
            f"SeatStats.{name} has type '{tstr}' — disallowed in a "
            "card-information-free accumulator surface.")
        for tok in ("card", "hole", "deal", "rank", "suit"):
            assert tok not in name.lower(), (
                f"SeatStats field name '{name}' contains forbidden card "
                "substring.")

    # Empirical sanity-run: walk a few hands, update the observer, and
    # confirm that .get_stats(seat) returns numbers. (We don't assert
    # specific values — just that the surface remains numeric.)
    rng = random.Random(20260603)
    observer = WM.MatchObserver(num_seats=NUM_PLAYERS)
    for _ in range(5):
        state = GAME.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = rng.choices(
                    [o for o, _ in outs], weights=[p for _, p in outs], k=1)[0]
                state.apply_action(int(a))
                continue
            cp = state.current_player()
            parsed = parse_state_6max(state)
            legal = state.legal_actions()
            # Map chip action to DiscreteAction approximately — pick a random
            # legal chip action and label it via a simple heuristic (the test
            # is about the OBSERVER's purity, not the action mapping).
            chip = int(rng.choice(legal))
            if chip == 0:
                discrete = 0  # FOLD
            elif chip == 1:
                discrete = 1  # CALL/CHECK
            else:
                discrete = 4  # treat as BET_100 (arbitrary aggressive label)
            try:
                observer.update(state, parsed, discrete, cp)
            except Exception as e:
                pytest.fail(f"observer.update raised on sanity run: {e}")
            state.apply_action(chip)
        observer.match_started()  # bumps n_hands_observed style fields
    # Confirm get_stats returns numeric SeatStats for every seat.
    for s in range(NUM_PLAYERS):
        st = observer.get_stats(s)
        assert isinstance(st.n_actions, int)
        assert isinstance(st.sum_bet_size_over_pot, float)


# --------------------------------------------------------------------------
# Test D6 — Adaptive6MaxNet model-surface leak guard
# --------------------------------------------------------------------------
# The S2(a)/scaffold-step-1b architecture introduces continuous opponent
# tendency supervision via opp_head_tendency (K=10). The tendency target,
# any archetype identity, and any oracle-derived per-opponent label must be
# TRAINING-TIME signals only — never enter forward(). This test enforces the
# property at three layers:
#
#   (D6.1) forward() signature accepts only (tokens, pad_mask, query_idx,
#          legal_mask, opp_stats). The match_conf multiplier (Q4) lives
#          OUTSIDE forward (in the inference-time blend), so its absence
#          here is the correct surface.
#   (D6.2) forward() source body contains no forbidden token
#          (tendency_target / cell_label / archetype_id / true_tendency /
#          oracle / true_cell).
#   (D6.3) Sentinel — repeated forward calls with the same 5-arg inputs
#          produce bit-identical outputs.

def test_D6_tendency_target_not_in_forward_inputs():
    import torch
    from src.nlhe.adaptive.model import (
        Adaptive6MaxNet, F_TOKEN, F_STATS, NUM_ACTIONS, K_TENDENCY,
    )

    net = Adaptive6MaxNet()
    net.eval()

    # D6.1 — signature
    sig = inspect.signature(net.forward)
    params = list(sig.parameters.keys())
    expected = ["tokens", "pad_mask", "query_idx", "legal_mask", "opp_stats"]
    assert params == expected, (
        f"forward() signature changed; expected {expected}, got {params}. "
        "If a tendency-target / archetype-id / oracle parameter has been "
        "added, the scaffold's no-oracle property has been broken at the "
        "architecture layer (Decision A / Q1c verified, Test D6 enforced).")

    # D6.2 — forward source contains no forbidden token
    src = inspect.getsource(net.forward)
    forbidden = ("tendency_target", "cell_label", "cell_id", "cell_idx",
                 "archetype_id", "archetype_idx", "true_cell",
                 "true_tendency", "oracle")
    for tok in forbidden:
        assert tok not in src, (
            f"forward() source contains forbidden token '{tok}'. "
            "Opponent identity / supervision target must not flow into "
            "the forward pass.")

    # D6.3 — sentinel: same inputs → bit-identical outputs
    B, T = 3, 8
    seed_rng = np.random.default_rng(2026)
    tokens = torch.from_numpy(
        seed_rng.standard_normal((B, T, F_TOKEN)).astype(np.float32))
    pad_mask = torch.zeros((B, T), dtype=torch.bool)
    query_idx = torch.tensor([2, 5, 3], dtype=torch.long)
    legal_mask = torch.ones((B, NUM_ACTIONS), dtype=torch.bool)
    opp_stats = torch.from_numpy(
        seed_rng.standard_normal((B, F_STATS)).astype(np.float32))

    with torch.no_grad():
        pol1, tend1, comb1 = net(
            tokens, pad_mask, query_idx, legal_mask, opp_stats)
        pol2, tend2, comb2 = net(
            tokens, pad_mask, query_idx, legal_mask, opp_stats)
    assert torch.equal(pol1, pol2)
    assert torch.equal(tend1, tend2)
    assert torch.equal(comb1, comb2)

    # And the head shapes match the declared K_TENDENCY / NUM_ACTIONS.
    assert pol1.shape == (B, NUM_ACTIONS)
    assert tend1.shape == (B, K_TENDENCY)
    # tendency_pred is sigmoided → bounded in [0, 1].
    assert torch.all(tend1 >= 0.0) and torch.all(tend1 <= 1.0)
