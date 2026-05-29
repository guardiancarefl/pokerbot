"""In-process worker for one or more parallel CFR traversals.

run_traversals(WorkerInput) rebuilds enough state from the input to call
traverse_6max with buffer writes redirected to local collectors, then
returns one WorkerOutput per assigned traversal_id. All randomness comes
from the explicit per-traversal fork rng_t = Random(seed*1_000_003 +
it*9_973 + t); no inherited globals are consulted. This is what makes
fork() safe for bit-identity (see DESIGN.md).

This module is callable in-process (for tests and single-worker debugging)
and is the unit a future orchestrator will fork().
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pyspiel
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.archetype6 import ArchetypePool
from src.nlhe.cfr6 import CFR6MaxContext, traverse_6max
from src.nlhe.checkpoint_registry import CheckpointRegistry
from src.nlhe.game_strings import TournamentStructure
from src.nlhe.infoset6 import InfosetEncoder6Max
from src.nlhe.league_pool import LeaguePool
from src.nlhe.networks6 import N_DISCRETE_ACTIONS
from src.nlhe.parallel.protocol import (
    TraversalSample,
    WorkerInput,
    WorkerOutput,
)
from src.nlhe.solver import MLP
from src.nlhe.solver6 import OVERRIDE_SALT, STACK_SAMPLE_SALT
from src.nlhe.stack_sampler import sample_starting_state


class _CollectorBuffer:
    """Drop-in for ReservoirBuffer.add(...) that appends to a list.
    Workers never call the real reservoir's add() — that would consume the
    buffer's own rng (seed+1 / seed+100), which lives on the orchestrator.
    """

    __slots__ = ("samples",)

    def __init__(self) -> None:
        self.samples: list[TraversalSample] = []

    def add(
        self,
        feature: np.ndarray,
        target: np.ndarray,
        legal_mask: np.ndarray,
        iteration: int,
    ) -> None:
        self.samples.append(TraversalSample(
            feature=feature,
            target=target,
            legal_mask=legal_mask,
            iteration=iteration,
        ))


class _NetsStub:
    """Minimal stand-in for PlayerNetworks6Max exposing exactly what
    traverse_6max touches: predict_advantages, buffer_for, strat_buffer.

    A fresh _NetsStub is constructed per traversal so each traversal's
    collectors start empty — but the loaded MLPs are SHARED across
    traversals in the same worker (rebuilding them once amortizes the
    state_dict load).
    """

    def __init__(self, nets: list[Any]) -> None:
        self._nets = nets
        self._adv_collector = _CollectorBuffer()
        self.strat_buffer = _CollectorBuffer()

    def predict_advantages(self, seat: int, features: np.ndarray) -> np.ndarray:
        # Mirrors PlayerNetworks6Max.predict_advantages bit-for-bit on CPU.
        x = torch.from_numpy(features).float().unsqueeze(0)
        with torch.no_grad():
            out = self._nets[seat](x)
        return out.squeeze(0).cpu().numpy()

    def buffer_for(self, seat: int) -> _CollectorBuffer:
        # traverse_6max only calls buffer_for(traversing_player) — the seat
        # arg is always the traverser within a given traversal, so a single
        # collector is sufficient. The seat is accepted for interface parity.
        return self._adv_collector


def _fork_rng(seed: int, iteration: int, t: int) -> random.Random:
    """Per-traversal RNG fork. Explicit integer arithmetic (NOT hash()) so
    the derivation is identical across processes — hash() randomization
    would break cross-process determinism. Matches solver6.py's in-tree
    derivation exactly.
    """
    s = (seed * 1_000_003 + iteration * 9_973 + t) & 0x7FFFFFFFFFFFFFFF
    return random.Random(s)


def _fork_rng_override(seed: int, iteration: int, t: int) -> random.Random:
    """Per-traversal override-RNG fork. Statistically independent stream from
    _fork_rng via OVERRIDE_SALT offset (imported from solver6). Worker side and
    orchestrator side derive identical rng_override_t values from the same
    (seed, iteration, t), so the band roll + pool sample agree without any
    cross-process communication.
    """
    s = (seed * 1_000_003 + iteration * 9_973 + t + OVERRIDE_SALT) & 0x7FFFFFFFFFFFFFFF
    return random.Random(s)


def _fork_rng_stack(seed: int, iteration: int, t: int) -> random.Random:
    """Per-traversal stack-sample RNG fork (Phase 3, tournament mode).
    Third independent stream via STACK_SAMPLE_SALT, matches solver6's
    tournament-branch derivation exactly so sequential and parallel agree
    on sample_starting_state's draws across the fork boundary.
    """
    s = (seed * 1_000_003 + iteration * 9_973 + t + STACK_SAMPLE_SALT) & 0x7FFFFFFFFFFFFFFF
    return random.Random(s)


def _build_nets(
    state_dicts: list[dict], input_dim: int, hidden_dim: list[int]
) -> list:
    nets = []
    for sd in state_dicts:
        net = MLP(in_dim=input_dim, hidden=hidden_dim, out_dim=N_DISCRETE_ACTIONS)
        net.load_state_dict(sd)
        net.eval()
        nets.append(net)
    return nets


def _sample_override_worker(
    rng: random.Random,
    league_pool,
    archetype_pool,
    archetype_mix: float,
    league_mix: float,
):
    """Worker-side band-roll + pool sampling. Mirrors solver6
    _maybe_sample_league_opponent's logic EXACTLY (same branches, same draw
    count per branch, same order) — bit-identity at mix>0 depends on this
    parity. Counter bookkeeping stays orchestrator-side (workers have no
    access to solver._override_counts across the fork boundary).
    """
    archetype_active = archetype_pool is not None and archetype_mix > 0.0
    league_active = league_pool is not None and league_mix > 0.0
    if not archetype_active and not league_active:
        return None
    r = rng.random()
    if r < archetype_mix:
        if archetype_pool is None:
            return None
        return archetype_pool.sample_opponent(rng)
    if r < archetype_mix + league_mix:
        if league_pool is None:
            return None
        return league_pool.sample_opponent(rng)
    return None


def _build_pools(wi: WorkerInput, abstraction):
    """Reconstruct league_pool and archetype_pool locally from WorkerInput.
    None paths → None pools (mix=0 short-circuit). Pools are constructed
    once per run_traversals call (= once per worker per iter); per-traversal
    sampling reuses them via their internal lazy caches.
    """
    league_pool = None
    if wi.league_registry_path is not None:
        registry = CheckpointRegistry.load(wi.league_registry_path)
        league_pool = LeaguePool(
            registry=registry,
            abstraction=abstraction,
            structure=None,  # Phase 2 stays legacy-mode-only
            sample_strategy=wi.league_sample_strategy,
            weights=wi.league_weights,
            recency_halflife=wi.league_recency_halflife,
            tag_filter=wi.league_tag_filter,
        )
    archetype_pool = None
    if wi.archetype_calibration_path is not None:
        archetype_pool = ArchetypePool(
            calibration_path=wi.archetype_calibration_path,
            abstraction=abstraction,
            profile_names=wi.archetype_profile_names,
            bucket_runouts=wi.encoder_bucket_runouts,
        )
    return league_pool, archetype_pool


def run_traversals(wi: WorkerInput) -> list[WorkerOutput]:
    """Execute the worker's assigned traversals; return tagged samples.

    All resources (game, abstraction, encoder, nets) are built locally from
    the WorkerInput — no parent objects are inherited via fork shared state.
    The encoder cache is shared across this worker's assigned traversals
    (within-worker memoization is safe because bucket lookups are pure
    functions of (hero, board); see DESIGN.md).
    """
    # Legacy-mode game is built once from wi.game_str. In tournament mode the
    # game spec varies per traversal (sampled stacks + dealer), so we defer
    # pyspiel.load_game to inside the per-traversal loop and build via
    # structure.to_inner_game_string_for_state(...). Loading wi.game_str
    # eagerly is harmless when tournament_structure is set (the result is
    # simply unused), so we keep it for code simplicity.
    game = pyspiel.load_game(wi.game_str)
    abstraction = Abstraction.load(wi.abstraction_path)
    encoder = InfosetEncoder6Max(
        abstraction=abstraction,
        starting_stack=wi.encoder_starting_stack,
        max_bucket_dim=wi.encoder_max_bucket_dim,
        bucket_runouts=wi.encoder_bucket_runouts,
    )
    nets = _build_nets(wi.adv_state_dicts, wi.input_dim, wi.hidden_dim)
    league_pool, archetype_pool = _build_pools(wi, abstraction)
    tournament_structure = None
    if wi.tournament_structure_path is not None:
        tournament_structure = TournamentStructure.from_yaml(
            wi.tournament_structure_path
        )

    outputs: list[WorkerOutput] = []
    for t in wi.traversal_ids:
        nets_stub = _NetsStub(nets)
        rng_t = _fork_rng(wi.seed, wi.iteration, t)
        rng_override_t = _fork_rng_override(wi.seed, wi.iteration, t)
        opp_override = _sample_override_worker(
            rng_override_t,
            league_pool,
            archetype_pool,
            wi.archetype_mix,
            wi.league_mix,
        )
        if tournament_structure is not None:
            # Tournament mode: sample per-traversal starting state, rebuild
            # the game from the sampled (stacks, dealer_seat, blind_level).
            rng_stack_t = _fork_rng_stack(wi.seed, wi.iteration, t)
            sampled = sample_starting_state(
                tournament_structure,
                rng_stack_t,
                num_paid=wi.num_paid,
            )
            gs = tournament_structure.to_inner_game_string_for_state(
                blind_level=sampled["blind_level"],
                stacks=sampled["stacks"],
                dealer_seat=sampled["dealer_seat"],
            )
            traversal_game = pyspiel.load_game(gs)
            state = traversal_game.new_initial_state()
            ctx = CFR6MaxContext(
                policy_nets=nets_stub,
                encoder=encoder,
                starting_stacks=list(sampled["stacks"]),
                payouts=wi.payouts,
                iteration=wi.iteration,
                max_depth=wi.max_depth,
                num_paid=wi.num_paid,
                dealer_seat=sampled["dealer_seat"],
            )
        else:
            # Legacy mode: reuse the cached game + WorkerInput's static stacks.
            state = game.new_initial_state()
            ctx = CFR6MaxContext(
                policy_nets=nets_stub,
                encoder=encoder,
                starting_stacks=wi.starting_stacks,
                payouts=wi.payouts,
                iteration=wi.iteration,
                max_depth=wi.max_depth,
                num_paid=wi.num_paid,
                dealer_seat=wi.dealer_seat,
            )
        traverse_6max(
            state,
            traversing_player=wi.traverser,
            ctx=ctx,
            rng=rng_t,
            opponent_policy_override=opp_override,
        )
        outputs.append(WorkerOutput(
            traversal_id=t,
            adv_samples=nets_stub._adv_collector.samples,
            strat_samples=nets_stub.strat_buffer.samples,
        ))
    return outputs
