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
from src.nlhe.cfr6 import CFR6MaxContext, traverse_6max
from src.nlhe.infoset6 import InfosetEncoder6Max
from src.nlhe.networks6 import N_DISCRETE_ACTIONS
from src.nlhe.parallel.protocol import (
    TraversalSample,
    WorkerInput,
    WorkerOutput,
)
from src.nlhe.solver import MLP


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


def run_traversals(wi: WorkerInput) -> list[WorkerOutput]:
    """Execute the worker's assigned traversals; return tagged samples.

    All resources (game, abstraction, encoder, nets) are built locally from
    the WorkerInput — no parent objects are inherited via fork shared state.
    The encoder cache is shared across this worker's assigned traversals
    (within-worker memoization is safe because bucket lookups are pure
    functions of (hero, board); see DESIGN.md).
    """
    game = pyspiel.load_game(wi.game_str)
    abstraction = Abstraction.load(wi.abstraction_path)
    encoder = InfosetEncoder6Max(
        abstraction=abstraction,
        starting_stack=wi.encoder_starting_stack,
        max_bucket_dim=wi.encoder_max_bucket_dim,
        bucket_runouts=wi.encoder_bucket_runouts,
    )
    nets = _build_nets(wi.adv_state_dicts, wi.input_dim, wi.hidden_dim)

    outputs: list[WorkerOutput] = []
    for t in wi.traversal_ids:
        nets_stub = _NetsStub(nets)
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
        rng_t = _fork_rng(wi.seed, wi.iteration, t)
        state = game.new_initial_state()
        traverse_6max(
            state,
            traversing_player=wi.traverser,
            ctx=ctx,
            rng=rng_t,
        )
        outputs.append(WorkerOutput(
            traversal_id=t,
            adv_samples=nets_stub._adv_collector.samples,
            strat_samples=nets_stub.strat_buffer.samples,
        ))
    return outputs
