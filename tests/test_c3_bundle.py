"""C3 retrain-bundle gates (2026-06-10).

Covers:
  - Step 2: encoder eff_stack_in_BB channel (236 -> 237), floors-convention
    match, legacy-layout byte-identity, old/new checkpoint dim dispatch.
  - Step 3: ante-convention stamping + live-serving resolution, and the
    clean-convention game-string sweep (correct blinds/antes at every
    level and alive-count; empty seats post nothing).
  - Step 5 plumbing: empirical-distribution sampling config validation +
    a one-iteration training smoke on a tiny artifact.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re

import numpy as np
import pyspiel
import pytest
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.conventions import (ANTE_INFLATED, ANTE_REAL,
                                  KNOWN_CKPT_CONVENTIONS,
                                  require_live_servable,
                                  resolve_ante_convention)
from src.nlhe.game_strings import TournamentStructure, six_max_sng
from src.nlhe.infoset6 import EFF_BB_SCALE, InfosetEncoder6Max, parse_state_6max
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max

ABSTRACTION_PATH = "runs/abstraction_20260521_223018_retrofit/abstraction.pkl"
STRUCTURE_PATH = "configs/ignition_double_up_6max_turbo.yaml"


@pytest.fixture(scope="module")
def abstraction():
    if not os.path.exists(ABSTRACTION_PATH):
        pytest.skip(f"abstraction not found at {ABSTRACTION_PATH}")
    return Abstraction.load(ABSTRACTION_PATH)


@pytest.fixture(scope="module")
def structure():
    return TournamentStructure.from_yaml(STRUCTURE_PATH)


def _state_for(structure, level, stacks, dealer_seat, rng_seed=7):
    gs = structure.to_inner_game_string_for_state(
        blind_level=structure.level(level), stacks=stacks,
        dealer_seat=dealer_seat)
    state = pyspiel.load_game(gs).new_initial_state()
    rng = random.Random(rng_seed)
    while state.is_chance_node():
        actions, probs = zip(*state.chance_outcomes())
        state.apply_action(rng.choices(actions, weights=probs, k=1)[0])
    return state


# ===== Step 2: encoder =====

def test_default_encoder_dim_is_236(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500)
    assert enc.feature_dim == 236


def test_eff_bb_encoder_dim_is_237(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500,
                             include_eff_bb=True)
    assert enc.feature_dim == 237


def test_legacy_prefix_byte_identical(abstraction, structure):
    """The first 236 features must be bit-for-bit the legacy encoding."""
    state = _state_for(structure, 3, [1500, 900, 2400, 1500, 700, 2000], 0)
    old = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500)
    new = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500,
                             include_eff_bb=True)
    rng = random.Random(11)
    f_old = old.encode(state, rng=random.Random(11))
    f_new = new.encode(state, rng=rng)
    assert f_old.shape == (236,)
    assert f_new.shape == (237,)
    np.testing.assert_array_equal(f_old, f_new[:236])


@pytest.mark.parametrize("level,stacks,dealer", [
    (1, [1500, 1500, 1500, 1500, 1500, 1500], 0),
    (3, [400, 2200, 1500, 0, 3000, 1900], 1),
    (5, [600, 0, 5200, 0, 2100, 1100], 2),
    (8, [900, 0, 7100, 0, 0, 1000], 5),
])
def test_eff_bb_matches_floors_convention(abstraction, structure, level,
                                           stacks, dealer):
    """Encoder channel == floors' _hero_eff_bb quantity / EFF_BB_SCALE,
    across levels and alive-counts."""
    from scripts.short_stack_floor_ab import _hero_eff_bb
    state = _state_for(structure, level, stacks, dealer)
    parsed = parse_state_6max(state)
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500,
                             include_eff_bb=True)
    feat = enc.encode_from_parsed(parsed, rng=random.Random(3))
    floors_eff_bb, floors_bb = _hero_eff_bb(parsed)
    assert floors_bb == structure.level(level).big_blind  # raw BB, real-ante
    assert feat[236] == pytest.approx(floors_eff_bb / EFF_BB_SCALE, abs=1e-7)
    assert feat[236] > 0.0


def test_eff_bb_zero_when_bb_unknown(abstraction):
    enc = InfosetEncoder6Max(abstraction=abstraction, starting_stack=1500,
                             include_eff_bb=True)
    parsed = {
        "num_players": 6, "street_idx": 0, "current_player": 2, "pot": 75,
        "money": [1450, 1400, 1500, 1500, 1500, 1500],
        "contribution": [50, 100, 0, 0, 0, 0],
        "private_cards": "", "public_cards": "", "sequences": "",
        # no big_blind key at all
    }
    feat = enc.encode_from_parsed(parsed)
    assert feat[236] == 0.0


# ===== Step 2: checkpoint dim dispatch =====

def _tiny_cfg(**over):
    base = dict(starting_stack=1500, big_blind=100, small_blind=50,
                hidden_dim=[8], n_iterations=1, traversals_per_iter=1,
                train_steps_per_iter=1, batch_size=2, buffer_capacity=16,
                bucket_runouts=4, seed=3)
    base.update(over)
    return TrainConfig6Max(**base)


def _make_solver(abstraction, **over):
    cfg = _tiny_cfg(**over)
    game = pyspiel.load_game(six_max_sng(starting_stack=cfg.starting_stack))
    return DeepCFR6MaxSolver(game=game, abstraction=abstraction, config=cfg)


def test_old_236_checkpoint_loads_untouched(abstraction, tmp_path):
    """A checkpoint whose config_dict predates the C3 fields (simulated by
    stripping them) must reconstruct a 236-d solver with identical params."""
    solver = _make_solver(abstraction)
    assert solver.encoder.feature_dim == 236
    p = tmp_path / "old.pt"
    solver.save_checkpoint(p, slim=True)
    ckpt = torch.load(str(p), weights_only=False)
    for k in ("encoder_eff_bb", "ante_convention", "empirical_dist_path"):
        ckpt["config_dict"].pop(k, None)        # simulate a pre-C3 ckpt
    torch.save(ckpt, str(p))

    from scripts.eval_6max_self_play import _load_solver
    loaded = _load_solver(str(p), abstraction, None)
    assert loaded.encoder.feature_dim == 236
    assert loaded.encoder.include_eff_bb is False
    assert loaded.policy_nets.state_dict()["input_dim"] == 236
    for (n1, t1), (n2, t2) in zip(
            solver.policy_nets.state_dict()["nets"][0].items(),
            loaded.policy_nets.state_dict()["nets"][0].items()):
        assert n1 == n2
        assert torch.equal(t1, t2)


def test_new_237_checkpoint_roundtrip(abstraction, tmp_path):
    solver = _make_solver(abstraction, encoder_eff_bb=True)
    assert solver.encoder.feature_dim == 237
    p = tmp_path / "new.pt"
    solver.save_checkpoint(p, slim=True)
    ckpt = torch.load(str(p), weights_only=False)
    assert ckpt["config_dict"]["encoder_eff_bb"] is True
    assert ckpt["config_dict"]["ante_convention"] == "real"

    from scripts.eval_6max_self_play import _load_solver
    loaded = _load_solver(str(p), abstraction, None)
    assert loaded.encoder.feature_dim == 237
    assert loaded.encoder.include_eff_bb is True


# ===== Step 3: convention resolution =====

def test_resolve_stamped_real(tmp_path):
    p = tmp_path / "a.pt"
    torch.save({"config_dict": {"ante_convention": "real"}}, str(p))
    assert resolve_ante_convention(p) == ANTE_REAL
    assert require_live_servable(p) == ANTE_REAL


def test_resolve_stamped_inflated_refused(tmp_path):
    p = tmp_path / "b.pt"
    torch.save({"config_dict": {"ante_convention": "inflated_bb"}}, str(p))
    assert resolve_ante_convention(p) == ANTE_INFLATED
    with pytest.raises(RuntimeError, match="inflated_bb"):
        require_live_servable(p)


def test_resolve_unstamped_unknown_refused(tmp_path):
    p = tmp_path / "c.pt"
    torch.save({"config_dict": {"starting_stack": 1500}}, str(p))
    assert resolve_ante_convention(p) is None
    with pytest.raises(RuntimeError, match="no ante_convention stamp"):
        require_live_servable(p)


def test_resolve_unstamped_whitelisted(tmp_path, monkeypatch):
    from src.nlhe import conventions
    p = tmp_path / "d.pt"
    torch.save({"config_dict": {}}, str(p))
    sha = conventions.sha256_of_file(p)
    monkeypatch.setitem(KNOWN_CKPT_CONVENTIONS, sha, ANTE_REAL)
    assert resolve_ante_convention(p) == ANTE_REAL
    assert require_live_servable(p) == ANTE_REAL


def test_deployed_checkpoint_is_whitelisted_and_servable():
    deployed = ("runs/k200_real_ante_20260605_225847_PRESERVED/"
                "ckpt_iter_1500.pt")
    if not os.path.exists(deployed):
        pytest.skip("deployed checkpoint not on this host")
    assert require_live_servable(deployed) == ANTE_REAL


# ===== Step 3: clean-convention game strings at every level/alive-count =====

_RE_BLIND = re.compile(r"blind=([\d ]+),")
_RE_ANTE = re.compile(r"ante=([\d ]+),")


@pytest.mark.parametrize("n_alive", [4, 5, 6])
def test_game_string_blinds_antes_all_levels(structure, n_alive):
    """Per-seat arrays under the real-ante convention: every alive seat
    posts exactly the level ante, busted/empty seats post NOTHING, SB/BB
    post the raw (un-inflated) blinds, pre-action dead money =
    sb + bb + n_alive*ante."""
    deep = 100_000  # deep stacks so no short-stack capping obscures the math
    for bl in structure.blind_schedule:
        stacks = [deep] * 6
        for i in range(6 - n_alive):       # bust trailing seats
            stacks[5 - i] = 0
        gs = structure.to_inner_game_string_for_state(
            blind_level=bl, stacks=stacks, dealer_seat=0)
        blinds = [int(x) for x in _RE_BLIND.search(gs).group(1).split()]
        antes = [int(x) for x in _RE_ANTE.search(gs).group(1).split()]
        alive = [i for i in range(6) if stacks[i] > 0]
        sb_seat = alive[1 % n_alive]
        bb_seat = alive[2 % n_alive]
        for i in range(6):
            if stacks[i] == 0:
                assert antes[i] == 0, f"L{bl.level}: busted seat {i} antes"
                assert blinds[i] == 0, f"L{bl.level}: busted seat {i} blinds"
            else:
                assert antes[i] == bl.ante, f"L{bl.level}: seat {i} ante"
        assert blinds[sb_seat] == bl.small_blind
        assert blinds[bb_seat] == bl.big_blind        # raw, NOT inflated
        assert sum(blinds) + sum(antes) == (
            bl.small_blind + bl.big_blind + n_alive * bl.ante)


def test_min_raise_l1_is_two_real_bb(structure):
    """The pre-train gate's check: L1 min raise-to = 2 x real BB = 50."""
    state = _state_for(structure, 1, [1500] * 6, 0)
    bl = structure.level(1)
    raise_actions = [a for a in state.legal_actions() if a >= 2]
    assert raise_actions, "no raise actions at L1 initial state"
    assert min(raise_actions) == 2 * bl.big_blind == 50


# ===== Step 5: empirical-distribution plumbing =====

def test_empirical_requires_structure():
    with pytest.raises(ValueError, match="tournament_structure_path"):
        _tiny_cfg(empirical_dist_path="/tmp/x.json")


def test_empirical_parallel_now_allowed():
    """The C3 deltas are wired into WorkerInput (2026-06-10): a config with
    encoder_eff_bb / empirical_dist_path AND parallel_groups > 0 must now
    construct cleanly. Equivalence is gated in tests/test_parallel_c3.py."""
    cfg = _tiny_cfg(encoder_eff_bb=True, parallel_groups=2,
                    empirical_dist_path="/tmp/x.json",
                    tournament_structure_path=STRUCTURE_PATH)
    assert cfg.parallel_groups == 2 and cfg.encoder_eff_bb


def test_empirical_sampling_one_iteration(abstraction, tmp_path):
    """Solver consumes a tiny artifact and completes one training iteration
    on the 237-d encoder + clean-ante path."""
    rows = [
        {"seed": 1, "hand": 1, "level": 1,
         "stacks": [1500, 1500, 1500, 1500, 1500, 1500], "dealer": 0,
         "n_alive": 6},
        {"seed": 1, "hand": 9, "level": 3,
         "stacks": [700, 2900, 0, 1500, 2400, 1500], "dealer": 4,
         "n_alive": 5},
        {"seed": 2, "hand": 14, "level": 4,
         "stacks": [3100, 4100, 0, 0, 1100, 700], "dealer": 0,
         "n_alive": 4},
    ]
    artifact = {"version": "training_dist_test", "n_games": 2,
                "hand_starts": rows}
    p = tmp_path / "dist.json.gz"
    p.write_bytes(gzip.compress(json.dumps(artifact).encode()))

    solver = _make_solver(
        abstraction,
        encoder_eff_bb=True,
        empirical_dist_path=str(p),
        tournament_structure_path=STRUCTURE_PATH,
        traversals_per_iter=3,
    )
    assert solver.encoder.feature_dim == 237
    assert len(solver.empirical_rows) == 3
    metrics = solver.train()
    assert len(metrics["iter"]) == 1
