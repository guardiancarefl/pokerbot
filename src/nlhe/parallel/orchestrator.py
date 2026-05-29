"""Orchestrator for parallel (or in-process-serial) CFR traversal.

Drives training using a real DeepCFR6MaxSolver's components but replaces
the traversal loop in solver6.train() with: extract adv-net state_dicts;
partition [0..T-1] into G groups; dispatch each group to run_traversals
(in-process or in a fork()ed worker); collect tagged WorkerOutputs; merge
them into the real buffers in strict ascending t order while interleaving
the orchestrator-side override sampler in the same order.

Bit-identity to solver6.train() at mix=0 requires that EVERY step
outside the parallelized traversal phase runs in the exact order the
sequential train() runs them in. See DESIGN.md and PHASE A of the build
recon for the operation list.

solver6.py is NOT modified; the orchestrator is a separate driver that
consumes the solver's public-ish components.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.nlhe.networks6 import NUM_SEATS_6MAX
from src.nlhe.parallel.protocol import (
    TraversalSample,
    WorkerInput,
    WorkerOutput,
)
from src.nlhe.parallel.worker import run_traversals
from src.nlhe.solver6 import DeepCFR6MaxSolver, OVERRIDE_SALT


def _extract_adv_state_dicts(solver: DeepCFR6MaxSolver) -> list[dict]:
    """Snapshot the 6 advantage nets' parameters. Detach + clone so the
    snapshot is independent of any later in-place training mutations.
    """
    return [
        {k: v.detach().clone() for k, v in solver.policy_nets.nets[s].state_dict().items()}
        for s in range(NUM_SEATS_6MAX)
    ]


def _partition_traversals(T: int, G: int) -> list[list[int]]:
    """Partition [0..T-1] into G contiguous groups (balanced as evenly as
    possible). Each traversal lands in exactly one group; groups preserve
    ascending order of traversal_ids — the merge phase will iterate the
    SORTED union of all groups' traversal_ids regardless of group structure,
    so partition layout does not affect correctness, only worker balance.
    """
    if G < 1:
        raise ValueError(f"G must be >= 1, got {G}")
    if G > T:
        G = T
    groups: list[list[int]] = [[] for _ in range(G)]
    for t in range(T):
        groups[t % G].append(t)
    return groups


def _build_worker_input(
    *,
    seed: int,
    iteration: int,
    traverser: int,
    traversal_ids: list[int],
    adv_state_dicts: list[dict],
    input_dim: int,
    hidden_dim: list[int],
    abstraction_path: str,
    encoder_starting_stack: int,
    encoder_max_bucket_dim: int,
    encoder_bucket_runouts: int,
    game_str: str,
    starting_stacks: list[int],
    payouts: list[float],
    max_depth: int,
    num_paid: int,
    dealer_seat: Optional[int],
    league_registry_path: Optional[str] = None,
    league_sample_strategy: str = "uniform",
    league_weights: Optional[dict] = None,
    league_recency_halflife: float = 5.0,
    league_tag_filter: Optional[list] = None,
    league_mix: float = 0.0,
    archetype_calibration_path: Optional[str] = None,
    archetype_profile_names: Optional[list] = None,
    archetype_mix: float = 0.0,
    tournament_structure_path: Optional[str] = None,
) -> WorkerInput:
    return WorkerInput(
        seed=seed,
        iteration=iteration,
        traverser=traverser,
        traversal_ids=traversal_ids,
        adv_state_dicts=adv_state_dicts,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        abstraction_path=abstraction_path,
        encoder_starting_stack=encoder_starting_stack,
        encoder_max_bucket_dim=encoder_max_bucket_dim,
        encoder_bucket_runouts=encoder_bucket_runouts,
        game_str=game_str,
        starting_stacks=starting_stacks,
        payouts=payouts,
        max_depth=max_depth,
        num_paid=num_paid,
        dealer_seat=dealer_seat,
        league_registry_path=league_registry_path,
        league_sample_strategy=league_sample_strategy,
        league_weights=league_weights,
        league_recency_halflife=league_recency_halflife,
        league_tag_filter=league_tag_filter,
        league_mix=league_mix,
        archetype_calibration_path=archetype_calibration_path,
        archetype_profile_names=archetype_profile_names,
        archetype_mix=archetype_mix,
        tournament_structure_path=tournament_structure_path,
    )


def _run_workers_inproc(worker_inputs: list[WorkerInput]) -> list[list[WorkerOutput]]:
    """Serial in-process execution: call run_traversals on each input.
    Used for Stage-1 validation (isolates merge logic from mp risk).
    """
    return [run_traversals(wi) for wi in worker_inputs]


def _run_workers_mp(
    worker_inputs: list[WorkerInput], n_workers: int
) -> list[list[WorkerOutput]]:
    """Stage-2 path: fork() per iter via mp.Pool.

    Each iter creates a fresh Pool that fork()s n_workers and dies on exit
    (spawn-per-iter semantics). Pool.map preserves result order matching
    input order, so group_0's outputs come back first, etc.
    """
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=n_workers) as pool:
        return pool.map(run_traversals, worker_inputs)


def _merge_outputs_into_buffers(
    solver: DeepCFR6MaxSolver,
    traverser: int,
    T: int,
    group_outputs: list[list[WorkerOutput]],
    seed: int,
    iteration: int,
) -> None:
    """Replay worker outputs into the real reservoir buffers in strict
    ASCENDING traversal_id order, interleaved with the orchestrator-side
    override-counter call so override counts advance in lock-step with
    sequential train(). Phase 2: count_only=True keeps the orchestrator
    off the pool-sampling cost (workers do the actual override sampling
    using the same deterministic rng_override_t derivation).
    """
    # Flatten across groups, then sort by traversal_id. Worker outputs
    # within a group are already in ascending t order (run_traversals
    # processes wi.traversal_ids in the given order), so the flatten is
    # a multi-list merge by id.
    by_id: dict[int, WorkerOutput] = {}
    for outputs in group_outputs:
        for o in outputs:
            if o.traversal_id in by_id:
                raise ValueError(
                    f"duplicate traversal_id {o.traversal_id} across groups"
                )
            by_id[o.traversal_id] = o
    if set(by_id.keys()) != set(range(T)):
        missing = set(range(T)) - set(by_id.keys())
        extra = set(by_id.keys()) - set(range(T))
        raise ValueError(
            f"partition is not exactly [0..{T-1}]: missing={missing} extra={extra}"
        )

    adv_buf = solver.policy_nets.buffer_for(traverser)
    strat_buf = solver.policy_nets.strat_buffer

    for t in range(T):
        # Counter bookkeeping: workers sampled the actual override using
        # rng_override_t locally; we derive the SAME rng here and call
        # _maybe_sample_league_opponent(rng=..., count_only=True) to
        # advance solver._override_counts deterministically without paying
        # the pool-sample cost (the Policy is discarded anyway since the
        # worker already traversed with its own override).
        rng_override_t = random.Random(
            (seed * 1_000_003 + iteration * 9_973 + t + OVERRIDE_SALT)
            & 0x7FFFFFFFFFFFFFFF
        )
        solver._maybe_sample_league_opponent(rng=rng_override_t, count_only=True)

        out = by_id[t]
        for s in out.adv_samples:
            adv_buf.add(s.feature, s.target, s.legal_mask, s.iteration)
        for s in out.strat_samples:
            strat_buf.add(s.feature, s.target, s.legal_mask, s.iteration)


def parallel_train(
    solver: DeepCFR6MaxSolver,
    *,
    game_str: str,
    abstraction_path: str,
    n_workers: int,
    use_processes: bool = False,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 10,
) -> dict:
    """Drop-in replacement for solver.train() with the traversal phase
    parallelized.

    Mirrors solver6.train()'s per-iter operation order exactly (see PHASE A
    recon in DESIGN.md); only the traversal loop is replaced. At mix=0 +
    legacy mode (no tournament_structure), this produces metrics
    bit-identical to solver.train() at the same seed (Stage-1 gate).

    Args:
        solver: an already-constructed DeepCFR6MaxSolver. Its
            policy_nets / encoder / starting_stacks / payouts / cfg /
            game / log / _override_counts are used in-place.
        game_str: the universal_poker string used to load solver.game.
            Workers re-load locally from this; pyspiel.Game does not pickle.
        abstraction_path: path to the abstraction pickle. Workers re-load
            locally; the Abstraction object does not cross processes.
        n_workers: group count G (== number of fork()ed worker processes
            per iter when use_processes=True; just the partition count when
            use_processes=False).
        use_processes: False = Stage-1 (in-process serial), True = Stage-2
            (fork()ed via mp.Pool).
        checkpoint_dir, checkpoint_every: same semantics as solver.train().

    Returns:
        Metrics dict with the same shape as solver.train()'s output.
    """
    cfg = solver.cfg
    # Phase 3 (tournament mode): supported. When solver.tournament_structure
    # is set, workers reload the TournamentStructure from
    # tournament_structure_path and derive rng_stack_t per traversal (mirrors
    # solver6's STACK_SAMPLE_SALT fork in the tournament branch of train()).
    # Each worker computes its own per-traversal starting_stacks and
    # dealer_seat; the WorkerInput.starting_stacks / dealer_seat fields are
    # unused in this mode but harmless.

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    T = cfg.traversals_per_iter
    G = n_workers

    start_iter = solver.iteration + 1
    metrics: dict = {
        "iter": [], "time": [], "traverser": [], "adv_loss": [],
        "strat_loss": [], "strat_buf": [], "mini_eval": [],
    }
    for s in range(NUM_SEATS_6MAX):
        metrics[f"buf_{s}"] = []

    runner = _run_workers_mp if use_processes else _run_workers_inproc

    t_start = time.time()
    for it in range(start_iter, cfg.n_iterations + 1):
        # ---- Steps 1-5: identical to solver6.train() ----
        solver.iteration = it
        traverser = (it - 1) % NUM_SEATS_6MAX
        t_it = time.time()
        solver._override_counts = {"archetype": 0, "league": 0, "self_play": 0}
        solver.encoder.reset_cache()

        # ---- Step 6/7: parallel traversal phase ----
        # Extract net snapshots BEFORE workers run; nets aren't mutated
        # during the worker phase (training is steps 8-9 below).
        adv_state_dicts = _extract_adv_state_dicts(solver)
        groups = _partition_traversals(T, G)
        worker_inputs = [
            _build_worker_input(
                seed=cfg.seed,
                iteration=it,
                traverser=traverser,
                traversal_ids=group,
                adv_state_dicts=adv_state_dicts,
                input_dim=solver.encoder.feature_dim,
                hidden_dim=list(cfg.hidden_dim),
                abstraction_path=abstraction_path,
                encoder_starting_stack=cfg.starting_stack,
                encoder_max_bucket_dim=solver.encoder.max_bucket_dim,
                encoder_bucket_runouts=cfg.bucket_runouts,
                game_str=game_str,
                starting_stacks=list(solver.starting_stacks),
                payouts=list(solver.payouts),
                max_depth=cfg.max_traversal_depth,
                num_paid=cfg.num_paid,
                dealer_seat=None,
                league_registry_path=cfg.league_registry_path,
                league_sample_strategy=cfg.league_sample_strategy,
                league_weights=cfg.league_weights,
                league_recency_halflife=cfg.league_recency_halflife,
                league_tag_filter=cfg.league_tag_filter,
                league_mix=cfg.league_mix,
                archetype_calibration_path=cfg.archetype_calibration_path,
                archetype_profile_names=cfg.archetype_profiles,
                archetype_mix=cfg.archetype_mix,
                tournament_structure_path=cfg.tournament_structure_path,
            )
            for group in groups
        ]
        if use_processes:
            group_outputs = _run_workers_mp(worker_inputs, len(groups))
        else:
            group_outputs = _run_workers_inproc(worker_inputs)

        _merge_outputs_into_buffers(
            solver, traverser, T, group_outputs,
            seed=cfg.seed, iteration=it,
        )

        # ---- Steps 8-14: identical to solver6.train() ----
        adv_loss = solver._train_advantage_net(traverser)
        strat_loss = solver._train_strategy_net()

        elapsed = time.time() - t_it
        metrics["iter"].append(it)
        metrics["time"].append(elapsed)
        metrics["traverser"].append(traverser)
        metrics["adv_loss"].append(adv_loss)
        metrics["strat_loss"].append(strat_loss)
        metrics["strat_buf"].append(len(solver.policy_nets.strat_buffer))
        for s in range(NUM_SEATS_6MAX):
            metrics[f"buf_{s}"].append(len(solver.policy_nets.buffer_for(s)))

        # Default-log path (matches solver6.train() default branch).
        solver.log(
            f"iter {it:>4}/{cfg.n_iterations}  "
            f"trav={traverser}  "
            f"adv={'nan' if math.isnan(adv_loss) else f'{adv_loss:.4f}':>8}  "
            f"strat={'nan' if math.isnan(strat_loss) else f'{strat_loss:.4f}':>8}  "
            f"bufs=({', '.join(str(len(solver.policy_nets.buffer_for(s))) for s in range(NUM_SEATS_6MAX))})  "
            f"sbuf={len(solver.policy_nets.strat_buffer)}  "
            f"{elapsed:.1f}s"
        )

        # Order mirrors sequential solver.train(): checkpoint first, then
        # mini_eval. The self-anchor lookup inside _maybe_run_mini_eval reads
        # the just-written checkpoint when it == mini_eval_every, so ordering
        # is load-bearing — do not swap.
        if checkpoint_dir is not None and (
            it % checkpoint_every == 0 or it == cfg.n_iterations
        ):
            ckpt_path = checkpoint_dir / f"ckpt_iter_{it:04d}.pt"
            solver.save_checkpoint(ckpt_path, slim=True)
            solver.log(f"  saved checkpoint: {ckpt_path}")

        # Mini-eval uses an isolated eval seed (seed+200+it) — does not touch
        # solver.rng or worker rngs. Runs on the main process between iters.
        if cfg.mini_eval_enabled and it % cfg.mini_eval_every == 0:
            solver._maybe_run_mini_eval(it, metrics, checkpoint_dir)

    total = time.time() - t_start
    solver.log(f"=== total: {total/60:.1f} min ===")
    return metrics
