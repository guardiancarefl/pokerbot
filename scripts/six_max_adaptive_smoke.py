"""6-max NLHE adaptive scaffold — CPU smoke (step 4 of the scaffold build).

This is a CORRECTNESS CHECK, not a real training run. Goals:
  - gradients flow on Adaptive6MaxNet (G1)
  - distill / aux losses decrease (G2/G3)
  - distill FLOOR (G4) — mean KL(hero_raw || blueprint) ≤ 0.02 nats AND
    p95 KL ≤ 0.05 nats over a fixed 1000-state eval set — at every checkpoint
  - total wall time ≤ 30 min (G5)
  - the head's tendency-vector READ separates archetypes from the blueprint
    (G6): D̄_blueprint near zero AND D̄_archetype / D̄_blueprint clearly > 1,
    ideally ≥ 2-3 (per Correction 2 — SEPARATION reasoning, not absolute bar)

ARCHITECTURE: smoke-only A4 size (d_model=64, num_layers=2, dim_ff=256).
This is the trunk-A shrunk-twice config that fits the 30-min G5 budget per
the wall-time benchmark (commit f94eb65). The DEPLOYED model size is an
OPEN GPU-phase tuning decision — NOT this A4 smoke-size. Per the user's
(D-explicit-at-call) ruling, the constructor defaults in
src/nlhe/adaptive/model.py STAY at d=128/L=4/ff=512 (design intent); this
script instantiates A4 explicitly at the call site below.

POOL COMPOSITION (per Correction 1 — oversample exploitables):
    40 matches  GusHansenMTT      (LAG, primary exploit target)
    35 matches  KillPhilMTT       (TAG, secondary exploit target)
    15 matches  Loosenluckymtt    (loose-passive, diversity)
    30 matches  blueprint         (negative control per Q5; load-bearing
                                    for G6 separation)
    ------------------
    120 matches total

LOSS WIRING (per Decision B — distill dominant, hard-locked):
    L = lam_distill · L_distill + lam_aux · L_aux
    lam_distill = 1.0       HARD-LOCKED (assert at start; --unsafe-override
                             flag exists but forbidden in smoke)
    lam_aux     = 0.10      the ONLY lever; first action on G4 floor breach
                             is HALVING this and restarting
    L_distill   = forward-KL(policy_raw, blueprint(state, hero_seat))
    L_aux       = MSE(tendency_pred, tendency_target)

READ PRIMITIVE (Q1c, head-derived, NO oracle):
    d_t = ||tendency_pred - tendency_blueprint||_2 in K=10 tendency space
    tendency_blueprint is the BLUEPRINT'S OWN expected tendency vector,
    measured ONCE at scaffold start via blueprint-vs-blueprint self-play.
    NOT opponent-specific.

ENCODE-ONCE BUFFER (per Mandatory Fix):
    Each public decision in a match is encoded EXACTLY ONCE per (match,
    hero_seat). The per-hand token buffer accumulates the encoder output
    at each decision; per-hero-decision training tuples reference slices
    of the buffer rather than re-encoding the prefix. Avoids the
    O(decisions²) naive re-encode trap.

BUILD ORDER (per user instruction):
    (1) write this script — DONE
    (2) MEASURE pool-gen cost on 5 matches (--measure-only flag) — HARD
        STOP if projected full-pool cost is over the budget
    (3) if under, run the full smoke

Run (measure-only):
    cd ~/pokerbot && .venv/bin/python -m scripts.six_max_adaptive_smoke \
        --anchor-dir runs/six_max_20260530_034023_phase4f_dcfr_candC_k200 \
        --abstraction-pkl runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
        --measure-only

Run (full smoke, after measurement OK):
    cd ~/pokerbot && setsid .venv/bin/python -m scripts.six_max_adaptive_smoke \
        --anchor-dir runs/six_max_20260530_034023_phase4f_dcfr_candC_k200 \
        --abstraction-pkl runs/abstraction_20260521_223018_retrofit/abstraction.pkl \
        > /tmp/sixmax_smoke.log 2>&1 < /dev/null & disown

§8/A6/B6/C6/D6 leakage preflight is the HARD gate before any pool gen.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pyspiel
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


# --------------------------------------------------------------------------
# §8/Test-A6/B6/C6/D6 preflight (HARD STOP)
# --------------------------------------------------------------------------

def gate_on_leakage_tests(test_file: str = "tests/test_six_max_token_no_leak.py"):
    print(f"[preflight] running pytest {test_file}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "--tb=short", "--no-header"],
        capture_output=True, text=True,
    )
    summary = proc.stdout.strip().split("\n")[-1]
    print(summary, flush=True)
    if proc.returncode != 0:
        print(proc.stdout, flush=True)
        print(proc.stderr, file=sys.stderr, flush=True)
        sys.exit("[preflight] §8 leakage tests FAILED — refusing to proceed.")
    print("[preflight] leakage tests GREEN — proceeding to model + scaffold",
          flush=True)


gate_on_leakage_tests()


# Safe to import everything else now.
from src.nlhe.game_strings import six_max_sng
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max
from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import DiscreteAction, discretize_legal_actions
from src.nlhe.cfr6 import _build_view_6max
from src.nlhe.infoset6 import parse_state_6max, InfosetEncoder6Max
from src.nlhe import within_match as WM
from src.nlhe.scripted_bots.policy import ShankyProfilePolicy
from src.nlhe.adaptive.model import (
    Adaptive6MaxNet, F_TOKEN, F_STATS, NUM_ACTIONS, K_TENDENCY,
    collate, tendency_l2_read,
)

GAME = pyspiel.load_game(six_max_sng())
NUM_SEATS = 6
NEG_INF = -1e9


def _log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Blueprint loader (the anchor)
# --------------------------------------------------------------------------

def load_blueprint(anchor_dir: Path, abstraction_pkl: Path):
    with open(abstraction_pkl, "rb") as f:
        absn: Abstraction = pickle.load(f)
    cfg_path = anchor_dir / "config.json"
    with open(cfg_path) as f:
        cfg_d = json.load(f)
    cfg = TrainConfig6Max(
        starting_stack=int(cfg_d["starting_stack"]),
        big_blind=int(cfg_d["big_blind"]),
        small_blind=int(cfg_d["small_blind"]),
        payout_mode=cfg_d["payout_mode"],
        buy_in=float(cfg_d["buy_in"]),
        first_share=float(cfg_d["first_share"]),
        hidden_dim=list(cfg_d["hidden_dim"]),
        bucket_runouts=int(cfg_d["bucket_runouts"]),
        tournament_structure_path=None,  # not needed for inference
    )
    solver = DeepCFR6MaxSolver(game=GAME, abstraction=absn, config=cfg)
    # Pick the iter_2000 checkpoint per Q3
    ckpt_path = anchor_dir / "checkpoints" / "ckpt_iter_2000.pt"
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    solver.policy_nets.load_state_dict(ckpt["policy_nets"])
    _log(f"[blueprint] loaded {ckpt_path}, schema="
         f"{solver.policy_nets.loaded_schema_version}")
    return solver


def blueprint_policy(solver, parsed: dict, state: Any) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (probs_over_9_discrete, legal_mask_bool[9], discrete_to_chip_dict).
    Probabilities are zero on illegal actions, sum to 1.0 on the legal subset."""
    seat = parsed["current_player"]
    feat = solver.encoder.encode_from_parsed(
        parsed, rng=random.Random(seat * 100003 + 7))  # deterministic
    view = _build_view_6max(state, parsed)
    discrete_to_chip = discretize_legal_actions(list(state.legal_actions()), view)
    legal_mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for d in discrete_to_chip:
        legal_mask[int(d)] = True
    probs = solver.policy_nets.inference_policy(seat, feat, legal_mask)
    return probs, legal_mask, discrete_to_chip


# --------------------------------------------------------------------------
# SeatStats → K=10 tendency vector projection
# --------------------------------------------------------------------------

def seat_stats_to_tendency_vector(s: WM.SeatStats,
                                   reference: Optional[np.ndarray] = None
                                   ) -> np.ndarray:
    """Project SeatStats counters to the K_TENDENCY=10 vector layout. Zero-
    denominator cells fall back to `reference` (if provided) or 0.5 neutral.
    All entries clamped to [0, 1]; mean_bet_size is clipped at 3.0 then
    divided by 3.0 (open question #6 default)."""
    fallback = (lambda i: float(reference[i])) if reference is not None else (lambda i: 0.5)
    v = np.zeros(K_TENDENCY, dtype=np.float32)
    # 0: VPIP
    v[0] = (s.n_preflop_voluntary / s.n_preflop_decisions
            if s.n_preflop_decisions > 0 else fallback(0))
    # 1: PFR
    v[1] = (s.n_preflop_raises / s.n_preflop_decisions
            if s.n_preflop_decisions > 0 else fallback(1))
    # 2-4: AF flop / turn / river
    for k, street_idx in enumerate((1, 2, 3), start=2):
        denom = s.n_postflop_decisions[street_idx]
        if denom > 0:
            v[k] = s.n_postflop_aggressive[street_idx] / denom
        else:
            v[k] = fallback(k)
    # 5-8: F2B preflop / flop / turn / river
    for k, street_idx in enumerate((0, 1, 2, 3), start=5):
        denom = s.n_facing_bet[street_idx]
        if denom > 0:
            v[k] = s.n_folds_facing_bet[street_idx] / denom
        else:
            v[k] = fallback(k)
    # 9: mean bet size (clipped [0, 3], normalized to [0, 1])
    if s.n_bet_size_samples > 0:
        raw = s.sum_bet_size_over_pot / s.n_bet_size_samples
        v[9] = max(0.0, min(3.0, raw)) / 3.0
    else:
        v[9] = fallback(9)
    return np.clip(v, 0.0, 1.0)


# --------------------------------------------------------------------------
# Match runner with encode-once buffer
# --------------------------------------------------------------------------

@dataclass
class _MatchResult:
    """Per-match output of `run_match_collect`."""
    hero_seat: int
    designated_opp_seat: int
    tuples: list[dict]                # per-hero-decision training tuples
    final_designated_opp_stats: WM.SeatStats   # post-match SeatStats
    n_hands: int
    n_decisions_total: int            # total public decisions across hands
    n_encoder_calls: int              # once per public decision per match (encode-once)


def _noised_blueprint_action(solver, parsed, state, epsilon, rng) -> int:
    """Hero plays noised-blueprint: ε-mix with uniform over legal."""
    probs, legal_mask, d_to_chip = blueprint_policy(solver, parsed, state)
    n_legal = int(legal_mask.sum())
    # ε-mix in discrete space
    p_explore = (1.0 - epsilon) * probs + epsilon * (
        legal_mask.astype(np.float64) / max(n_legal, 1))
    p_explore = p_explore / p_explore.sum()
    discrete_action = int(rng.choices(
        range(NUM_ACTIONS), weights=list(p_explore), k=1)[0])
    # Map back to chip action
    chip = d_to_chip.get(DiscreteAction(discrete_action))
    if chip is None:
        # Shouldn't happen since we masked to legal; safety fallback
        chip = next(iter(d_to_chip.values()))
    return int(chip), int(discrete_action), legal_mask, probs, d_to_chip


def _shanky_to_discrete(action_chip: int, d_to_chip: dict) -> int:
    """Reverse lookup: chip action → DiscreteAction (best-effort)."""
    for d, c in d_to_chip.items():
        if int(c) == int(action_chip):
            return int(d)
    return int(DiscreteAction.CALL)  # safety


def run_match_collect(solver, opp_policy, hero_seat: int,
                      hand_target: int, epsilon: float,
                      designated_opp_seat: int,
                      seed: int, hero_encoder_rng_seed: int = 0):
    """Run one match (up to hand_target hands), record per-hero-decision
    tuples with the encode-once buffer for tokens, return the result."""
    py_rng = random.Random(seed)
    enc_rng = random.Random(hero_encoder_rng_seed)
    observer = WM.MatchObserver(num_seats=NUM_SEATS)
    observer.match_started()

    tuples = []
    n_hands = 0
    n_decisions_total = 0
    n_encoder_calls = 0

    for hand_i in range(hand_target):
        state = GAME.new_initial_state()
        n_hands += 1
        # Per-hand encode-once buffer (resets each hand). Each entry is the
        # 236-dim encoder output from HERO's perspective at one public
        # decision in this hand. We append BEFORE the actor decides, so the
        # transformer's query position (hero's current decision) is at the
        # last index of the buffer when we record a hero tuple.
        hand_buffer: list[np.ndarray] = []

        while not state.is_terminal():
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = py_rng.choices(
                    [o for o, _ in outs],
                    weights=[p for _, p in outs], k=1)[0]
                state.apply_action(int(a))
                continue
            cur = state.current_player()
            # PUBLIC DECISION: encode the state from HERO's POV ONCE
            # (encode-once buffer). The encoder's bucket cache is reset per
            # hand by the cache being keyed on (hero_cards, board); the rng
            # is for postflop MC fallback only.
            parsed_hero = parse_state_6max(state, observer=hero_seat)
            solver.encoder.reset_cache()
            tok_vec = solver.encoder.encode_from_parsed(
                parsed_hero, rng=enc_rng)
            hand_buffer.append(tok_vec.astype(np.float32))
            n_encoder_calls += 1
            n_decisions_total += 1

            # Cap buffer at 50 (T≤50 truncation; keep the last 50 only)
            if len(hand_buffer) > 50:
                hand_buffer = hand_buffer[-50:]

            # Now decide whose turn it is
            if cur == hero_seat:
                # HERO: noised-blueprint
                # Re-parse with default observer (= cur = hero) for blueprint
                # call's encoder cache. blueprint_policy parses internally.
                parsed_cur = parse_state_6max(state)
                chip, discrete, legal_mask, blueprint_probs, _ = \
                    _noised_blueprint_action(solver, parsed_cur, state,
                                              epsilon, py_rng)
                # Record HERO TUPLE: tokens = current hand_buffer (up to and
                # including this hero decision, materialize a COPY).
                opp_stats_vec = seat_stats_to_tendency_vector(
                    observer.get_stats(designated_opp_seat))
                # F_STATS = 12: K_TENDENCY (10) + match_conf (1) + n_actions
                # normalized (1)
                match_conf = observer.confidence(designated_opp_seat)
                n_act = float(observer.get_stats(
                    designated_opp_seat).n_actions)
                opp_stats_full = np.concatenate([
                    opp_stats_vec,
                    np.array([match_conf, min(1.0, n_act / 300.0)],
                             dtype=np.float32)
                ])
                assert opp_stats_full.shape == (F_STATS,)
                # distill_target: blueprint at HERO's spot; zeros on illegal
                distill_t = blueprint_probs.astype(np.float32).copy()
                # Materialize tokens
                tokens = np.stack(hand_buffer).astype(np.float32)
                tuples.append({
                    "tokens": tokens,
                    "query_idx": int(tokens.shape[0] - 1),
                    "legal_mask": legal_mask.copy(),
                    "opp_stats": opp_stats_full,
                    "distill_target": distill_t,
                    "tendency_target": None,  # filled at match end
                    "hero_seat": int(hero_seat),
                    "designated_opp_seat": int(designated_opp_seat),
                    "match_conf_at_decision": float(match_conf),
                })
                # Update observer with hero's action (for completeness; the
                # designated-opp counters won't fire on hero's seat)
                observer.update(state, parsed_cur, discrete, hero_seat)
                state.apply_action(chip)
            else:
                # OPP: archetype select_action
                parsed_cur = parse_state_6max(state)
                view = _build_view_6max(state, parsed_cur)
                d_to_chip = discretize_legal_actions(
                    list(state.legal_actions()), view)
                chip_action = opp_policy.select_action(
                    parsed_cur, state, py_rng, mode="sample")
                discrete = _shanky_to_discrete(chip_action, d_to_chip)
                # Update observer for opp's seat
                observer.update(state, parsed_cur, discrete, cur)
                state.apply_action(int(chip_action))
        # end of hand

    # Fill tendency_target on each tuple with the post-match SeatStats
    final_stats = observer.get_stats(designated_opp_seat)
    tendency_target = seat_stats_to_tendency_vector(final_stats)
    for t in tuples:
        t["tendency_target"] = tendency_target.copy()

    return _MatchResult(
        hero_seat=hero_seat,
        designated_opp_seat=designated_opp_seat,
        tuples=tuples,
        final_designated_opp_stats=final_stats,
        n_hands=n_hands,
        n_decisions_total=n_decisions_total,
        n_encoder_calls=n_encoder_calls,
    )


# --------------------------------------------------------------------------
# Blueprint-as-opponent (for the negative control AND for tendency_blueprint)
# --------------------------------------------------------------------------

class _BlueprintAsOpp:
    """Adapter so the blueprint can act as an opponent in run_match_collect."""
    name = "blueprint"
    def __init__(self, solver):
        self.solver = solver

    def select_action(self, parsed, state, rng, mode="sample") -> int:
        probs, legal_mask, d_to_chip = blueprint_policy(self.solver, parsed, state)
        # Sample DiscreteAction proportional to probs (already masked legal)
        keys = np.flatnonzero(legal_mask).tolist()
        ws = [float(probs[k]) for k in keys]
        total = sum(ws)
        if total <= 0:
            choice = rng.choice(keys)
        else:
            choice = rng.choices(keys, weights=ws, k=1)[0]
        chip = d_to_chip.get(DiscreteAction(int(choice)))
        if chip is None:
            chip = next(iter(d_to_chip.values()))
        return int(chip)


# --------------------------------------------------------------------------
# Tendency blueprint reference (the universal GTO read reference)
# --------------------------------------------------------------------------

def measure_blueprint_tendency_reference(solver, n_matches: int = 20,
                                          hand_target: int = 30,
                                          seed: int = 9001) -> np.ndarray:
    """Run `n_matches` blueprint-vs-blueprint matches; for each match,
    collect the seat-1 (designated opp position) SeatStats; return the
    mean tendency vector. This is tendency_blueprint — the universal GTO
    reference for the Q1c L2 read primitive."""
    bp_opp = _BlueprintAsOpp(solver)
    refs = []
    for i in range(n_matches):
        # All seats play blueprint; rotate hero_seat for coverage but use
        # seat-1 designated for stats (irrelevant: any seat works under
        # symmetry, but consistent for reproducibility).
        hero_seat = i % NUM_SEATS
        # Use a thin wrapper as opponent (blueprint plays the opps).
        # Hero ALSO plays blueprint (epsilon=0.0 for the reference measure).
        result = run_match_collect(
            solver=solver, opp_policy=bp_opp, hero_seat=hero_seat,
            hand_target=hand_target, epsilon=0.0,
            designated_opp_seat=(hero_seat + 1) % NUM_SEATS,
            seed=seed + 100 * i, hero_encoder_rng_seed=i)
        refs.append(seat_stats_to_tendency_vector(
            result.final_designated_opp_stats))
    refs = np.stack(refs)
    return refs.mean(axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Pool composition (per Correction 1 — oversample exploitables)
# --------------------------------------------------------------------------

POOL_DEFAULTS = {
    "gushansenmtt":   {"path": "data/shanky_profiles/GusHansenMTT.txt",
                       "n_matches": 40},
    "killphilmtt":    {"path": "data/shanky_profiles/KillPhilMTT.txt",
                       "n_matches": 35},
    "loosenluckymtt": {"path": "data/shanky_profiles/Loosenluckymtt.txt",
                       "n_matches": 15},
    "blueprint":      {"path": None,           # special: blueprint itself
                       "n_matches": 30},
}


def make_opp_policy(name: str, info: dict, solver):
    if name == "blueprint":
        return _BlueprintAsOpp(solver)
    return ShankyProfilePolicy(name=name, profile_path=info["path"])


def generate_pool(solver, opp_specs: dict, hand_target: int,
                   epsilon: float, base_seed: int, verbose: bool = True):
    """Run all matches, return (pool_tuples, per_opp_results, stats)."""
    pool = []
    per_opp_results = {}
    t0_all = time.time()
    n_total_matches = sum(s["n_matches"] for s in opp_specs.values())
    match_i = 0
    for opp_name, info in opp_specs.items():
        opp_pol = make_opp_policy(opp_name, info, solver)
        per_opp_results[opp_name] = []
        for m in range(info["n_matches"]):
            hero_seat = m % NUM_SEATS
            designated_opp_seat = (hero_seat + 1) % NUM_SEATS
            t0 = time.time()
            result = run_match_collect(
                solver=solver, opp_policy=opp_pol, hero_seat=hero_seat,
                hand_target=hand_target, epsilon=epsilon,
                designated_opp_seat=designated_opp_seat,
                seed=base_seed + match_i * 17,
                hero_encoder_rng_seed=match_i)
            dt = time.time() - t0
            # Tag each tuple with opp_name for later attribution
            for t in result.tuples:
                t["opp_name"] = opp_name
            pool.extend(result.tuples)
            per_opp_results[opp_name].append(result)
            match_i += 1
            if verbose:
                _log(f"  [match {match_i}/{n_total_matches}] "
                     f"{opp_name:<16} hero={hero_seat} hands={result.n_hands} "
                     f"decisions={result.n_decisions_total} "
                     f"tuples={len(result.tuples)} ({dt:.1f}s)")
    dt_total = time.time() - t0_all
    stats = {
        "n_matches_total": n_total_matches,
        "n_pool_tuples": len(pool),
        "wall_seconds": dt_total,
        "ms_per_match": (1000 * dt_total / max(1, n_total_matches)),
    }
    return pool, per_opp_results, stats


# --------------------------------------------------------------------------
# Fixed eval set for floor check (G4)
# --------------------------------------------------------------------------

def build_eval_set(solver, n_states: int = 1000, seed: int = 2026):
    """Build a fixed eval set of (tokens, query_idx, legal_mask, opp_stats,
    blueprint_target_probs) tuples — same shape as training tuples but
    NOT used for training. Used for G4 floor checks."""
    py_rng = random.Random(seed)
    enc_rng = random.Random(seed + 1)
    eval_tuples = []
    while len(eval_tuples) < n_states:
        # Random hero seat per match; play one hand at a time and record
        # the hero's state via the same encode-once buffer pattern.
        hero_seat = py_rng.randrange(NUM_SEATS)
        state = GAME.new_initial_state()
        hand_buffer = []
        while not state.is_terminal() and len(eval_tuples) < n_states:
            if state.is_chance_node():
                outs = state.chance_outcomes()
                a = py_rng.choices(
                    [o for o, _ in outs],
                    weights=[p for _, p in outs], k=1)[0]
                state.apply_action(int(a))
                continue
            parsed_hero = parse_state_6max(state, observer=hero_seat)
            solver.encoder.reset_cache()
            tok = solver.encoder.encode_from_parsed(
                parsed_hero, rng=enc_rng).astype(np.float32)
            hand_buffer.append(tok)
            if len(hand_buffer) > 50:
                hand_buffer = hand_buffer[-50:]
            cur = state.current_player()
            if cur == hero_seat:
                # Record an eval tuple
                parsed_cur = parse_state_6max(state)
                probs, legal_mask, d_to_chip = blueprint_policy(
                    solver, parsed_cur, state)
                # opp_stats: dummy (eval is for distill floor only; opp_stats
                # not used by policy_head path mathematically but must be
                # valid shape). Use zeros for the SeatStats portion.
                opp_stats = np.zeros(F_STATS, dtype=np.float32)
                tokens = np.stack(hand_buffer).astype(np.float32)
                eval_tuples.append({
                    "tokens": tokens,
                    "query_idx": int(tokens.shape[0] - 1),
                    "legal_mask": legal_mask.copy(),
                    "opp_stats": opp_stats,
                    "blueprint_probs": probs.astype(np.float32),
                })
                # Step state with uniform-legal action (eval is at decision
                # node only; don't matter what action we apply)
                chip = py_rng.choice(list(d_to_chip.values()))
                state.apply_action(int(chip))
            else:
                # Step opp with uniform-legal chip action
                chip = py_rng.choice(state.legal_actions())
                state.apply_action(int(chip))
    return eval_tuples


def floor_check(net, eval_tuples, device: str,
                eps_mean: float, eps_p95: float, batch_size: int = 64
                ) -> tuple[bool, float, float]:
    """Run net on each eval state at gate=0 (just the raw policy_head),
    compute KL(net_raw || blueprint) per state, return (ok, mean, p95)."""
    net.eval()
    kls = []
    eps_tiny = 1e-12
    with torch.no_grad():
        for i in range(0, len(eval_tuples), batch_size):
            batch = eval_tuples[i:i+batch_size]
            packed = collate(batch, device=device)
            policy_raw, _, _ = net(
                packed["tokens"], packed["pad_mask"], packed["query_idx"],
                packed["legal_mask"], packed["opp_stats"])
            net_probs = policy_raw.cpu().numpy()
            for j, t in enumerate(batch):
                lm = t["legal_mask"]
                bp = t["blueprint_probs"]
                p = np.clip(net_probs[j][lm], eps_tiny, 1.0)
                p = p / p.sum()
                q = np.clip(bp[lm], eps_tiny, 1.0)
                q = q / q.sum()
                kls.append(float(np.sum(p * (np.log(p) - np.log(q)))))
    mean = float(np.mean(kls))
    p95 = float(np.quantile(kls, 0.95))
    ok = (mean <= eps_mean) and (p95 <= eps_p95)
    return ok, mean, p95


# --------------------------------------------------------------------------
# G6 probe (per Correction 2 — SEPARATION reasoning, not absolute bar)
# --------------------------------------------------------------------------

def probe_g6(net, solver, opp_specs, tendency_blueprint, hand_target,
             n_probe_matches_per_opp: int, base_seed: int):
    """For each opp type, run probe matches; at hero decisions, query the
    net for tendency_pred; compute per-opp mean D̄ = ||tendency_pred -
    tendency_blueprint||_2. Report absolute D̄ per opp AND the ratio
    D̄_archetype / D̄_blueprint (the separation signal)."""
    net.eval()
    results = {}
    for opp_name, info in opp_specs.items():
        opp_pol = make_opp_policy(opp_name, info, solver)
        d_bars = []
        cosine_proxy_tendencies = []
        for m in range(n_probe_matches_per_opp):
            hero_seat = m % NUM_SEATS
            designated_opp_seat = (hero_seat + 1) % NUM_SEATS
            result = run_match_collect(
                solver=solver, opp_policy=opp_pol, hero_seat=hero_seat,
                hand_target=hand_target, epsilon=0.0,
                designated_opp_seat=designated_opp_seat,
                seed=base_seed + m * 17,
                hero_encoder_rng_seed=m + 1000)
            # Run net on every hero decision in this match; collect
            # tendency_pred; average.
            if not result.tuples:
                continue
            with torch.no_grad():
                packed = collate(result.tuples, device="cpu")
                _, tendency_pred, _ = net(
                    packed["tokens"], packed["pad_mask"],
                    packed["query_idx"], packed["legal_mask"],
                    packed["opp_stats"])
            tpred = tendency_pred.cpu().numpy()
            for t in tpred:
                d_bars.append(tendency_l2_read(t, tendency_blueprint))
            # Take the average tendency at hand-end-equivalent (last 5
            # decisions) for cosine separation reporting
            cosine_proxy_tendencies.append(np.mean(tpred[-5:], axis=0))
        results[opp_name] = {
            "n_probe_matches": n_probe_matches_per_opp,
            "n_decisions_total": len(d_bars),
            "mean_d_bar": float(np.mean(d_bars)) if d_bars else float("nan"),
            "median_d_bar": float(np.median(d_bars)) if d_bars else float("nan"),
            "mean_tendency_at_hand_end": (
                np.mean(np.stack(cosine_proxy_tendencies), axis=0).tolist()
                if cosine_proxy_tendencies else [float("nan")]*K_TENDENCY),
        }
    return results


# --------------------------------------------------------------------------
# Cotrain loop
# --------------------------------------------------------------------------

def cotrain(net, pool, eval_set, *, epochs: int, steps_per_epoch: int,
            batch_size: int, lr: float, lam_distill: float, lam_aux: float,
            eps_mean: float, eps_p95: float, ckpt_every: int,
            device: str, seed: int, out_dir: Path):
    """Train net on the pool with the distill + tendency-MSE loss.
    Floor (G4) checked every `ckpt_every` epochs; HARD STOP on breach."""
    # Decision-B assertion
    assert lam_distill == 1.0, (
        f"lam_distill is HARD-LOCKED at 1.0 in smoke; got {lam_distill}. "
        "Use --unsafe-override-distill-weight to override (forbidden in smoke).")

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(seed)
    log = []
    train_kls_distill = []
    train_mses_aux = []
    floor_checks = []
    net.train()
    n_pool = len(pool)

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        ep_L_distill = 0.0
        ep_L_aux = 0.0
        n_steps = 0
        for s in range(steps_per_epoch):
            idx = rng.choice(n_pool, size=batch_size, replace=False)
            batch = [pool[i] for i in idx]
            packed = collate(batch, device=device)
            distill_target = torch.from_numpy(np.stack(
                [b["distill_target"] for b in batch])).to(device)
            tendency_target = torch.from_numpy(np.stack(
                [b["tendency_target"] for b in batch])).to(device)
            policy_raw, tendency_pred, _ = net(
                packed["tokens"], packed["pad_mask"], packed["query_idx"],
                packed["legal_mask"], packed["opp_stats"])
            log_p = torch.log(policy_raw + 1e-12)
            L_distill = -(distill_target * log_p).sum(dim=-1).mean()
            L_aux = F.mse_loss(tendency_pred, tendency_target)
            L = lam_distill * L_distill + lam_aux * L_aux
            opt.zero_grad()
            L.backward()
            # G1: gradient-flow sanity (check on the first step only — cheap)
            if ep == 1 and s == 0:
                bad = False
                grad_norm_sq = 0.0
                for p in net.parameters():
                    if p.grad is None:
                        bad = True; break
                    g = p.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        bad = True; break
                    grad_norm_sq += float((g ** 2).sum())
                if bad:
                    sys.exit("[G1 FAIL] gradients have NaN/Inf or some param "
                             "received no gradient — STOP.")
                if grad_norm_sq <= 0:
                    sys.exit("[G1 FAIL] gradient norm is zero — STOP.")
                _log(f"  [G1 OK] grad norm = {math.sqrt(grad_norm_sq):.4f}")
            opt.step()
            ep_L_distill += float(L_distill.detach())
            ep_L_aux += float(L_aux.detach())
            n_steps += 1
        dt = time.time() - t_ep
        log.append({
            "epoch": ep,
            "L_distill": ep_L_distill / n_steps,
            "L_aux": ep_L_aux / n_steps,
            "seconds": dt,
        })
        train_kls_distill.append(ep_L_distill / n_steps)
        train_mses_aux.append(ep_L_aux / n_steps)
        if ep == 1 or ep % max(1, epochs // 20) == 0 or ep == epochs:
            _log(f"  [cotrain ep {ep:>3d}/{epochs}] "
                 f"L_d={ep_L_distill/n_steps:.4f} "
                 f"L_a={ep_L_aux/n_steps:.4f} ({dt:.1f}s)")
        # G4 floor check at ckpt_every
        if ep % ckpt_every == 0 or ep == epochs:
            ok, m_kl, p95_kl = floor_check(
                net, eval_set, device=device,
                eps_mean=eps_mean, eps_p95=eps_p95)
            floor_checks.append({
                "epoch": ep, "ok": ok, "mean_kl": m_kl, "p95_kl": p95_kl,
                "eps_mean": eps_mean, "eps_p95": eps_p95,
            })
            verdict = "OK" if ok else "BREACH"
            _log(f"    [G4 floor ep {ep}] mean_KL={m_kl:.4f} "
                 f"(bar={eps_mean}) p95_KL={p95_kl:.4f} "
                 f"(bar={eps_p95}) -> {verdict}")
            if not ok:
                _log(f"    [G4 BREACH] STOP per Decision B; first lever = "
                     f"halve --lam-aux (currently {lam_aux}) and restart.")
                return {"log": log, "floor_checks": floor_checks,
                        "stopped_early_at_epoch": ep,
                        "g4_breach": True,
                        "train_kls_distill": train_kls_distill,
                        "train_mses_aux": train_mses_aux}
            net.train()
        if ep % ckpt_every == 0:
            torch.save({"state_dict": net.state_dict(), "epoch": ep},
                       out_dir / f"ckpt_ep{ep:04d}.pt")
    return {"log": log, "floor_checks": floor_checks,
            "stopped_early_at_epoch": None,
            "g4_breach": False,
            "train_kls_distill": train_kls_distill,
            "train_mses_aux": train_mses_aux}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--abstraction-pkl", required=True)
    ap.add_argument("--hand-target", type=int, default=40,
                    help="hands per match (cap)")
    ap.add_argument("--epsilon", type=float, default=0.20,
                    help="noised-blueprint hero ε (uniform mix)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--steps-per-epoch", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-distill", type=float, default=1.0,
                    help="HARD-LOCKED at 1.0 in smoke per Decision B.")
    ap.add_argument("--lam-aux", type=float, default=0.10,
                    help="ONLY lever per Decision B. Halve on G4 breach.")
    ap.add_argument("--eps-floor-mean", type=float, default=0.02,
                    help="G4 mean KL bar; measured on blueprint baseline.")
    ap.add_argument("--eps-floor-p95", type=float, default=0.05,
                    help="G4 p95 KL bar; tail catch.")
    ap.add_argument("--n-eval-states", type=int, default=1000)
    ap.add_argument("--n-blueprint-ref-matches", type=int, default=20)
    ap.add_argument("--n-probe-matches-per-opp", type=int, default=8)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=None)
    ap.add_argument("--measure-only", action="store_true",
                    help="Run a small-sample pool-gen cost measurement and "
                         "EXIT (per the user's MANDATORY FIX before the "
                         "full smoke).")
    ap.add_argument("--measure-n-matches-each", type=int, default=2,
                    help="In --measure-only, matches per archetype "
                         "(total = 4 * this).")
    # Trunk-size diagnostics (smoke-only override; not the deployed size)
    ap.add_argument("--d-model", type=int, default=64,
                    help="Trunk d_model. Default 64 = smoke-A4 (benchmarked "
                         "to fit cotrain budget). Bump to 128 (design-intent "
                         "default) for pure-distill capacity diagnostics; "
                         "FULL cotrain at d=128 blows the 30-min budget.")
    ap.add_argument("--num-layers", type=int, default=2,
                    help="Trunk num_layers. Default 2 = smoke-A4.")
    ap.add_argument("--dim-ff", type=int, default=256,
                    help="Trunk dim_ff. Default 256 = smoke-A4.")
    ap.add_argument("--unsafe-override-distill-weight",
                    action="store_true",
                    help="FORBIDDEN in smoke — would unlock --lam-distill "
                         "from its HARD-LOCK at 1.0. Asserted off.")
    args = ap.parse_args()
    if args.unsafe_override_distill_weight:
        sys.exit("--unsafe-override-distill-weight is FORBIDDEN in smoke.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.out or
               f"runs/sixmax_smoke_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    _log(f"out_dir = {out}")

    # 1. Load blueprint
    t0 = time.time()
    solver = load_blueprint(Path(args.anchor_dir),
                             Path(args.abstraction_pkl))
    _log(f"[load] blueprint loaded in {time.time()-t0:.1f}s; "
         f"encoder.feature_dim={solver.encoder.feature_dim}")

    if args.measure_only:
        _log("")
        _log("=== 5-MATCH POOL-GEN COST MEASUREMENT (HARD GATE) ===")
        opp_specs = {
            "gushansenmtt": {"path": POOL_DEFAULTS["gushansenmtt"]["path"],
                              "n_matches": args.measure_n_matches_each},
            "killphilmtt": {"path": POOL_DEFAULTS["killphilmtt"]["path"],
                             "n_matches": args.measure_n_matches_each},
            "loosenluckymtt": {"path": POOL_DEFAULTS["loosenluckymtt"]["path"],
                                "n_matches": args.measure_n_matches_each},
            "blueprint": {"path": None,
                           "n_matches": args.measure_n_matches_each},
        }
        n_sample = sum(s["n_matches"] for s in opp_specs.values())
        _log(f"[measure] sampling {n_sample} matches "
             f"({args.measure_n_matches_each} per archetype incl. blueprint)")
        _, _, stats = generate_pool(
            solver=solver, opp_specs=opp_specs,
            hand_target=args.hand_target, epsilon=args.epsilon,
            base_seed=args.seed, verbose=True)
        full_total_matches = sum(POOL_DEFAULTS[k]["n_matches"] for k in POOL_DEFAULTS)
        proj_min = (stats["ms_per_match"] / 1000 * full_total_matches) / 60
        _log("")
        _log(f"[measure RESULT] {stats['n_matches_total']} matches in "
             f"{stats['wall_seconds']:.1f}s; "
             f"~{stats['ms_per_match']:.0f} ms/match; "
             f"pool tuples generated: {stats['n_pool_tuples']}")
        _log(f"[measure PROJECTION] full pool ({full_total_matches} matches) "
             f"≈ {proj_min:.1f} min wall time")
        if proj_min > 18.0:
            _log(f"[measure VERDICT] OVER BUDGET — projected {proj_min:.1f} "
                 f"min exceeds the ~18-min pool-gen budget (leaves ~12 min "
                 f"for training + floor + G6 within the 30-min G5 ceiling). "
                 f"STOP for user review per Mandatory Fix.")
        else:
            _log(f"[measure VERDICT] UNDER BUDGET — projected {proj_min:.1f} "
                 f"min fits within the ~18-min pool-gen budget. SAFE to "
                 f"proceed to full smoke.")
        _log("")
        _log("[measure-only] exiting. Re-run without --measure-only for the "
             "full smoke after user OK.")
        return

    # --- FULL SMOKE PATH BELOW (only reachable after measure-only confirms) ---

    # 2. Build the universal tendency_blueprint reference
    _log("")
    t0 = time.time()
    tendency_blueprint = measure_blueprint_tendency_reference(
        solver, n_matches=args.n_blueprint_ref_matches,
        hand_target=args.hand_target, seed=args.seed + 9001)
    _log(f"[tendency_blueprint] measured in {time.time()-t0:.1f}s")
    _log(f"  layout = (VPIP, PFR, AF_flop, AF_turn, AF_river, "
         f"F2B_pf, F2B_f, F2B_t, F2B_r, mean_bet/3)")
    _log(f"  values = {[f'{v:.3f}' for v in tendency_blueprint.tolist()]}")

    # 3. Pool generation
    _log("")
    _log(f"[pool gen] composition (per Correction 1 — oversample exploitables):")
    opp_specs = {k: {"path": v["path"], "n_matches": v["n_matches"]}
                 for k, v in POOL_DEFAULTS.items()}
    for k, v in opp_specs.items():
        _log(f"  {k:<16} {v['n_matches']} matches")
    pool, per_opp_results, pool_stats = generate_pool(
        solver=solver, opp_specs=opp_specs, hand_target=args.hand_target,
        epsilon=args.epsilon, base_seed=args.seed, verbose=False)
    _log(f"[pool gen] DONE: {pool_stats['n_pool_tuples']} tuples from "
         f"{pool_stats['n_matches_total']} matches in "
         f"{pool_stats['wall_seconds']:.1f}s "
         f"(~{pool_stats['ms_per_match']:.0f} ms/match)")

    # 4. Build fixed eval set for floor check
    _log("")
    t0 = time.time()
    eval_set = build_eval_set(solver, n_states=args.n_eval_states,
                               seed=args.seed + 12345)
    _log(f"[eval-set] {len(eval_set)} fixed states in {time.time()-t0:.1f}s")

    # 5. Instantiate Adaptive6MaxNet at the configured trunk size
    # (SMOKE-ONLY per (D-explicit-at-call) ruling). The constructor defaults
    # in src/nlhe/adaptive/model.py STAY at d=128/L=4/ff=512 (design intent);
    # this script sets the trunk size explicitly via CLI flags.
    net = Adaptive6MaxNet(
        d_model=args.d_model, num_layers=args.num_layers,
        nhead=4, dim_ff=args.dim_ff)
    n_params = sum(p.numel() for p in net.parameters())
    _log(f"[net] Adaptive6MaxNet (d={args.d_model}, L={args.num_layers}, "
         f"ff={args.dim_ff}) — {n_params:,} params")
    _log(f"      smoke-only size for CPU diagnostics; real training size "
         f"is a GPU-phase tuning decision (commit f94eb65)")

    # 5b. PRE-TRAINING floor sanity (verify metric)
    ok, m_pre, p95_pre = floor_check(
        net, eval_set, device="cpu",
        eps_mean=args.eps_floor_mean, eps_p95=args.eps_floor_p95)
    _log(f"[pre-train floor sanity] random-init mean_KL={m_pre:.4f} "
         f"p95_KL={p95_pre:.4f} (expected large since untrained)")

    # 6. Cotrain
    _log("")
    _log("=== Cotrain ===")
    train_t0 = time.time()
    train_result = cotrain(
        net, pool, eval_set,
        epochs=args.epochs, steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size, lr=args.lr,
        lam_distill=args.lam_distill, lam_aux=args.lam_aux,
        eps_mean=args.eps_floor_mean, eps_p95=args.eps_floor_p95,
        ckpt_every=args.ckpt_every,
        device="cpu", seed=args.seed, out_dir=out)
    train_wall = time.time() - train_t0
    _log(f"[cotrain DONE] {train_wall:.1f}s wall "
         f"(epochs ran: "
         f"{train_result['stopped_early_at_epoch'] or args.epochs})")

    # 7. G2/G3 verdict
    L_d0 = train_result["train_kls_distill"][0]
    L_dN = train_result["train_kls_distill"][-1]
    L_a0 = train_result["train_mses_aux"][0]
    L_aN = train_result["train_mses_aux"][-1]
    G2_ok = L_dN < L_d0
    G3_ok = L_aN < L_a0
    _log(f"[G2] L_distill: ep1={L_d0:.4f}, epLast={L_dN:.4f} -> "
         f"{'OK' if G2_ok else 'FAIL'}")
    _log(f"[G3] L_aux:     ep1={L_a0:.4f}, epLast={L_aN:.4f} -> "
         f"{'OK' if G3_ok else 'FAIL (diagnostic only)'}")

    # 8. G4 summary
    G4_ok = all(c["ok"] for c in train_result["floor_checks"])
    _log(f"[G4] floor checks ({len(train_result['floor_checks'])} total) -> "
         f"{'OK' if G4_ok else 'BREACH'}")
    for c in train_result["floor_checks"]:
        _log(f"     ep{c['epoch']:>3d}: mean={c['mean_kl']:.4f} "
             f"p95={c['p95_kl']:.4f} -> "
             f"{'OK' if c['ok'] else 'BREACH'}")

    # 9. G6 probe (per Correction 2 — separation reasoning)
    _log("")
    _log("=== G6 probe (separation reasoning per Correction 2) ===")
    probe_t0 = time.time()
    g6 = probe_g6(net, solver, opp_specs, tendency_blueprint,
                   hand_target=args.hand_target,
                   n_probe_matches_per_opp=args.n_probe_matches_per_opp,
                   base_seed=args.seed + 88888)
    probe_dt = time.time() - probe_t0
    _log(f"[G6 probe] done in {probe_dt:.1f}s "
         f"(n_probe_matches_per_opp={args.n_probe_matches_per_opp})")
    bp_dbar = g6["blueprint"]["mean_d_bar"]
    _log(f"  blueprint D̄ (absolute, negative control): {bp_dbar:.4f}")
    _log(f"  per-opp D̄ + separation ratio vs blueprint:")
    sep_table = []
    for opp_name, r in g6.items():
        d = r["mean_d_bar"]
        ratio = d / bp_dbar if bp_dbar > 0 else float("inf")
        _log(f"    {opp_name:<16} D̄={d:.4f}  ratio_vs_bp={ratio:.2f}")
        sep_table.append((opp_name, d, ratio))
    # G6 SEPARATION verdict
    G6_blueprint_low = bp_dbar <= 0.05
    G6_separation_ok = all(
        r["mean_d_bar"] / max(bp_dbar, 1e-12) >= 2.0
        for n, r in g6.items() if n != "blueprint")
    _log(f"[G6] blueprint absolute (≤ 0.05 expected): "
         f"{bp_dbar:.4f} -> {'OK' if G6_blueprint_low else 'LOOSE'}")
    _log(f"[G6] separation ratio (all archetypes/blueprint ≥ 2.0): -> "
         f"{'OK' if G6_separation_ok else 'WEAK'}")

    # 10. Save artifacts + final report
    _log("")
    total_wall = time.time() - train_t0 + (
        pool_stats["wall_seconds"]) + 0  # approx
    metrics = {
        "phase": "sixmax_adaptive_smoke",
        "anchor_dir": args.anchor_dir,
        "abstraction_pkl": args.abstraction_pkl,
        "trunk": {
            "label": "A4_smoke_only",
            "d_model": 64, "num_layers": 2, "nhead": 4, "dim_ff": 256,
            "n_params": n_params,
        },
        "args": vars(args),
        "tendency_blueprint": tendency_blueprint.tolist(),
        "pool_stats": pool_stats,
        "train_log": train_result["log"],
        "floor_checks": train_result["floor_checks"],
        "G2_ok": bool(G2_ok), "G3_ok": bool(G3_ok),
        "G4_ok": bool(G4_ok), "G5_wall_seconds": train_wall,
        "G6": {"per_opp": g6,
               "blueprint_d_bar": bp_dbar,
               "blueprint_low_ok": bool(G6_blueprint_low),
               "separation_ok": bool(G6_separation_ok)},
        "g4_breach": bool(train_result["g4_breach"]),
        "stopped_early_at_epoch": train_result["stopped_early_at_epoch"],
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                  default=str))
    torch.save({"state_dict": net.state_dict(),
                "config": {"d_model": 64, "num_layers": 2, "nhead": 4,
                            "dim_ff": 256, "smoke_only": True}},
               out / "smoke_net.pt")
    _log(f"[saved] {out}/metrics.json + smoke_net.pt")
    _log(f"[STOP] smoke complete. STOP for user review.")


if __name__ == "__main__":
    main()
