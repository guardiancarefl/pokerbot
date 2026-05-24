"""6-max Deep CFR training loop (Phase 4e.3c).

Composes the pieces shipped in 4a/b/d/e.1/e.2/e.3a/e.3b:
  - game_strings.PokerGameConfig         (4a) — game string builder
  - icm.sng_payouts_6max_*               (4b) — payout structures
  - infoset6.InfosetEncoder6Max          (4d) — 236-dim features
  - trajectory6 (independent primitive)  (4e.1) — not used here (cfr6 walks tree directly)
  - icm_returns.icm_adjust_returns       (4e.2) — applied inside cfr6
  - networks6.PlayerNetworks6Max         (4e.3a) — 6 advantage nets + buffers
  - cfr6.traverse_6max + CFR6MaxContext  (4e.3b) — the traversal primitive

This module is the 6-max equivalent of the HUNL src/nlhe/solver.py's
training-loop portion. The traversal primitive lives in cfr6.py so it can
be tested in isolation; this file orchestrates iterations, trains advantage
nets from buffers, handles checkpointing.

Mirrors the HUNL pattern with these intentional 6-max-specific differences:

  1. Six advantage nets instead of two. Each iteration traverses for ONE
     seat (cycling: traverser = (it - 1) % 6). Only that seat's net trains
     per iteration. After 6 iterations, every seat has trained once.

  2. No strategy net yet. PlayerNetworks6Max (4e.3a) carries only advantage
     nets; average-strategy approximation is a Phase 4e.4 concern. At
     deployment time the policy is the regret-matched current strategy
     from the latest advantage net.

  3. No DCFR weighting yet. Vanilla CFR (uniform sample weights). DCFR
     for 6-max is mechanical to add once the baseline trains and we see
     a baseline trajectory.

  4. No archetype mix yet. Archetype framework (src/nlhe/archetypes.py)
     uses HUNL-specific position derivation; porting to 6-max is a
     separate subphase.

  5. Uniform starting stacks per traversal. Real SNG hands have evolving
     stacks across hands. First-cut 6-max training treats every traversal
     as a fresh hand with equal stacks ([cfg.starting_stack] * 6). ICM
     transformation still computes correctly per terminal, but the bot
     won't see bubble-pressure asymmetry from differing stacks. Stack
     distribution sampling is its own future subphase.

  6. Regrets NOT divided by starting_stack (see cfr6.py docstring): ICM
     utilities are already on O(1) equity scale.

Checkpoint format: torch.save'd dict containing PlayerNetworks6Max's
state_dict() (nets + optimizers in one shot), per-seat buffer state,
iteration counter, Python and torch RNG states, and a copy of the
config. The HUNL Session-8 bit-identical-resume invariant carries over:
saving at iter N and loading into a fresh solver produces parameters and
buffer contents indistinguishable from the original.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from src.nlhe.abstraction import Abstraction
from src.nlhe.cfr6 import CFR6MaxContext, traverse_6max
from src.nlhe.icm import (
    sng_payouts_6max_double_up,
    sng_payouts_6max_standard,
)
from src.nlhe.infoset6 import InfosetEncoder6Max
from src.nlhe.networks6 import (
    N_DISCRETE_ACTIONS,
    NUM_SEATS_6MAX,
    PlayerNetworks6Max,
)


# ===== Payout-mode resolution =====


def _resolve_payouts(payout_mode: str, buy_in: float, first_share: float) -> list[float]:
    """Map a payout_mode string to a payouts list.

    Args:
        payout_mode: 'double_up' or 'standard' (matches Ignition 6-max formats).
        buy_in: tournament buy-in per player (defines prize pool = 6 * buy_in).
        first_share: only used for 'standard' mode (default 0.65).

    Returns:
        Payouts list to feed CFR6MaxContext.
    """
    if payout_mode == "double_up":
        return sng_payouts_6max_double_up(buy_in=buy_in)
    if payout_mode == "standard":
        return sng_payouts_6max_standard(buy_in=buy_in, first_share=first_share)
    raise ValueError(
        f"unknown payout_mode={payout_mode!r}; expected 'double_up' or 'standard'"
    )


# ===== Config =====


@dataclass
class TrainConfig6Max:
    """Hyperparameters + game parameters for 6-max Deep CFR training.

    Fields ordered: game shape, training hyperparams, solver behavior.

    Args:
        starting_stack: per-seat chip count at the start of every traversal.
        big_blind / small_blind: blind structure.
        payout_mode: 'double_up' (Ignition top-3 equal) or 'standard'
            (Ignition top-2 65/35). Drives the ICM transformation.
        buy_in: per-player buy-in for prize pool sizing. The absolute
            number is arbitrary; only ratios matter for ICM.
        first_share: 1st-place prize share for 'standard' mode (ignored
            otherwise). 0.65 = industry-standard 65/35 split.

        hidden_dim: per-seat MLP hidden layers.
        n_iterations: total CFR iterations to run.
        traversals_per_iter: external-sampling traversals per iteration
            (all with the same traverser, cycled across iterations).
        train_steps_per_iter: gradient steps on the traverser's advantage
            net per iteration.
        batch_size: SGD batch size from the reservoir buffer.
        learning_rate: Adam LR per net.
        buffer_capacity: per-seat reservoir buffer capacity.
        bucket_runouts: MC runouts for postflop bucket lookups (encoder).
        max_traversal_depth: safety cap on recursion (healthy games << this).
        seed: random seed for Python + torch RNGs.
    """
    # Game shape.
    starting_stack: int = 1500
    big_blind: int = 100
    small_blind: int = 50
    payout_mode: str = "double_up"
    buy_in: float = 1.0
    first_share: float = 0.65

    # Training hyperparameters.
    hidden_dim: list[int] = field(default_factory=lambda: [256, 256])
    n_iterations: int = 100
    traversals_per_iter: int = 100
    train_steps_per_iter: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    buffer_capacity: int = 100_000

    # Encoder + solver behavior.
    bucket_runouts: int = 50
    max_traversal_depth: int = 500

    # Tournament-aware training. When set, each trajectory samples a
    # starting state via stack_sampler.sample_starting_state and builds
    # a fresh universal_poker game (rotated dealer, varied stacks, etc.).
    # When None (default), the legacy fixed-game path runs.
    tournament_structure_path: Optional[str] = None
    num_paid: int = 3

    # DCFR variants (Brown & Sandholm 2019). 'vanilla' preserves
    # pre-DCFR behavior bit-for-bit. 'linear' weights each sample
    # by t_i / T; 'discounted' uses (t_i / T) ** dcfr_exponent.
    # Per-sample iteration tag is recorded by cfr6.traverse_6max
    # via buf.add(..., iteration=ctx.iteration).
    cfr_variant: str = "vanilla"
    dcfr_exponent: float = 1.0

    # League play (Pillar 5). When league_mix > 0 AND league_registry_path
    # is set, the solver loads a CheckpointRegistry + LeaguePool and, with
    # probability league_mix per traversal, replaces self-play opponents
    # with a sampled league checkpoint for that whole traversal. Granularity
    # is per-traversal (all 5 opponent seats share one league policy within
    # one hand of the tree) — this matches how a league checkpoint was
    # trained and keeps integration simple. Per-seat-per-traversal sampling
    # is a future extension.
    #
    # Defaults preserve pre-league behavior bit-for-bit (league_mix=0.0 means
    # the override is never sampled and traverse_6max sees opponent_policy_
    # override=None on every call).
    league_mix: float = 0.0
    league_registry_path: Optional[str] = None
    league_sample_strategy: str = "uniform"
    league_recency_halflife: float = 5.0
    league_weights: Optional[dict] = None
    league_tag_filter: Optional[list] = None

    seed: int = 2026


# ===== Solver =====


class DeepCFR6MaxSolver:
    """Trainer for 6-max NLHE Deep CFR.

    Construction is decoupled from game and abstraction creation (matching
    the HUNL solver pattern): the caller builds those and passes them in.
    """

    def __init__(
        self,
        game: Any,
        abstraction: Abstraction,
        config: TrainConfig6Max,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if game.num_players() != NUM_SEATS_6MAX:
            raise ValueError(
                f"DeepCFR6MaxSolver requires a {NUM_SEATS_6MAX}-player game; "
                f"got {game.num_players()}"
            )

        self.game = game
        self.abstraction = abstraction
        self.cfg = config
        self.log = logger or print

        # Reproducibility: seed both Python and torch RNGs.
        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)

        # Encoder shared across all seats. Reset its bucket cache between
        # iterations to bound memory.
        self.encoder = InfosetEncoder6Max(
            abstraction=abstraction,
            starting_stack=config.starting_stack,
            max_bucket_dim=200,
            bucket_runouts=config.bucket_runouts,
        )

        # Six advantage networks. PlayerNetworks6Max owns nets, optimizers,
        # buffers — one set per seat.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_nets = PlayerNetworks6Max(
            input_dim=self.encoder.feature_dim,
            hidden=list(config.hidden_dim),
            learning_rate=config.learning_rate,
            buffer_capacity=config.buffer_capacity,
            rng=random.Random(config.seed + 1),
            device=self.device,
        )

        # Resolve payouts once at construction (config-fixed across training).
        self.payouts = _resolve_payouts(
            payout_mode=config.payout_mode,
            buy_in=config.buy_in,
            first_share=config.first_share,
        )

        # Uniform stacks for every traversal (see module docstring note 5).
        self.starting_stacks = [config.starting_stack] * NUM_SEATS_6MAX

        # Tournament-aware training: load the structure if configured.
        self.tournament_structure = None
        if config.tournament_structure_path is not None:
            from src.nlhe.game_strings import TournamentStructure
            self.tournament_structure = TournamentStructure.from_yaml(
                config.tournament_structure_path
            )
            self.log(
                f"  tournament mode: loaded "
                f"{self.tournament_structure.format_name} "
                f"({len(self.tournament_structure.blind_schedule)} levels)"
            )

        # League play: load registry + construct pool when both fields set.
        # league_mix == 0 with a registry path is allowed (loads pool but
        # never samples) — useful for debugging. league_mix > 0 with no
        # registry path is a configuration error.
        self.league_pool = None
        if config.league_mix > 0.0 and not config.league_registry_path:
            raise ValueError(
                "league_mix > 0 requires league_registry_path to be set; "
                f"got league_mix={config.league_mix}, "
                f"league_registry_path={config.league_registry_path!r}"
            )
        if config.league_registry_path:
            from src.nlhe.checkpoint_registry import CheckpointRegistry
            from src.nlhe.league_pool import LeaguePool
            registry = CheckpointRegistry.load(config.league_registry_path)
            self.league_pool = LeaguePool(
                registry=registry,
                abstraction=abstraction,
                structure=self.tournament_structure,
                sample_strategy=config.league_sample_strategy,
                weights=config.league_weights,
                recency_halflife=config.league_recency_halflife,
                tag_filter=config.league_tag_filter,
            )
            eligible = len(self.league_pool)
            if eligible == 0:
                raise ValueError(
                    "league_registry_path resolved to an empty eligible "
                    f"pool (registry={config.league_registry_path!r}, "
                    f"tag_filter={config.league_tag_filter!r})"
                )
            self.log(
                f"  league pool: {eligible} eligible checkpoints, "
                f"strategy={config.league_sample_strategy}, "
                f"mix={config.league_mix:.3f}"
            )

        self.iteration = 0

        self.log(
            f"DeepCFR6MaxSolver  device={self.device}  "
            f"feature_dim={self.encoder.feature_dim}  "
            f"payout_mode={config.payout_mode}  payouts={self.payouts}"
        )

    # ---- Network training ----

    def _maybe_sample_league_opponent(self):
        """Sample a league opponent for this traversal, or None.

        Returns:
            A Policy (CheckpointPolicy per LeaguePool) when the per-traversal
            league mix coin flip fires, or None when self-play should be used.

        Defaults preserve pre-league behavior: league_mix=0.0 (and/or
        league_pool=None) means this returns None on every call, so the
        downstream traverse_6max sees opponent_policy_override=None.
        """
        if self.league_pool is None:
            return None
        if self.cfg.league_mix <= 0.0:
            return None
        if self.rng.random() >= self.cfg.league_mix:
            return None
        return self.league_pool.sample_opponent(self.rng)

    def _dcfr_weights(self, iters):
        """Per-sample DCFR weights for a batch.

        Vanilla    -> None (caller falls back to unweighted mean).
        Linear     -> w_i = t_i / T
        Discounted -> w_i = (t_i / T) ** dcfr_exponent

        Weights are normalized to sum to batch_size, preserving gradient
        scale relative to the unweighted case. T defaults to max(1,
        current iter) to keep weights bounded in [0, 1] during iter 1.

        Mirrors HUNL src/nlhe/solver.py::_dcfr_weights verbatim.
        """
        variant = self.cfg.cfr_variant
        if variant == "vanilla":
            return None
        if variant not in ("linear", "discounted"):
            raise ValueError(
                f"unknown cfr_variant={variant!r}; expected vanilla|linear|discounted"
            )
        T = max(1, self.iteration)
        ratio = iters.float() / T
        if variant == "linear":
            w = ratio
        else:
            w = ratio ** self.cfg.dcfr_exponent
        s = w.sum().clamp(min=1e-8)
        return w * (w.shape[0] / s)

    def _train_advantage_net(self, seat: int) -> float:
        """One iteration of training on seat's advantage net.

        Returns mean MSE loss across `train_steps_per_iter` batches, or NaN
        if the buffer is smaller than batch_size (matches HUNL convention).
        """
        buf = self.policy_nets.buffer_for(seat)
        if len(buf) < self.cfg.batch_size:
            return float("nan")

        net = self.policy_nets.net_for(seat)
        opt = self.policy_nets.optimizer_for(seat)
        net.train()

        total_loss = 0.0
        for _ in range(self.cfg.train_steps_per_iter):
            feats, targets, masks, iters = buf.sample_batch(self.cfg.batch_size)
            feats = feats.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            iters = iters.to(self.device)

            preds = net(feats)
            # MSE on legal-action subset, summed across actions, per-sample,
            # then DCFR-weighted (vanilla -> None -> unweighted mean).
            per_sample = ((preds - targets) ** 2 * masks).sum(dim=1)
            weights = self._dcfr_weights(iters)
            loss = (weights * per_sample).mean() if weights is not None else per_sample.mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        net.eval()
        return total_loss / self.cfg.train_steps_per_iter

    # ---- Training loop ----

    def train(
        self,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_every: int = 10,
    ) -> dict:
        """Run the full 6-max CFR training loop.

        Args:
            checkpoint_dir: if provided, save a checkpoint every
                `checkpoint_every` iterations and on completion.
            checkpoint_every: cadence for periodic checkpoints.

        Returns:
            Metrics dict with per-iteration lists: iter, time, traverser,
            adv_loss (for the traversed seat), per-seat buffer sizes.
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_iter = self.iteration + 1
        metrics: dict = {
            "iter": [], "time": [], "traverser": [], "adv_loss": [],
        }
        for s in range(NUM_SEATS_6MAX):
            metrics[f"buf_{s}"] = []

        t_start = time.time()
        for it in range(start_iter, self.cfg.n_iterations + 1):
            self.iteration = it
            traverser = (it - 1) % NUM_SEATS_6MAX
            t_it = time.time()

            # Reset the encoder's per-game bucket cache to bound memory.
            self.encoder.reset_cache()

            # Build the per-iteration context once.
            if self.tournament_structure is None:
                # Legacy fixed-game mode.
                ctx = CFR6MaxContext(
                    policy_nets=self.policy_nets,
                    encoder=self.encoder,
                    starting_stacks=self.starting_stacks,
                    payouts=self.payouts,
                    iteration=it,
                    max_depth=self.cfg.max_traversal_depth,
                    num_paid=self.cfg.num_paid,
                )
                for _ in range(self.cfg.traversals_per_iter):
                    state = self.game.new_initial_state()
                    opp_override = self._maybe_sample_league_opponent()
                    traverse_6max(
                        state,
                        traversing_player=traverser,
                        ctx=ctx,
                        rng=self.rng,
                        opponent_policy_override=opp_override,
                    )
            else:
                # Tournament-aware mode: sample state + build game per trajectory.
                from src.nlhe.stack_sampler import sample_starting_state
                import pyspiel
                for _ in range(self.cfg.traversals_per_iter):
                    sampled = sample_starting_state(
                        self.tournament_structure,
                        self.rng,
                        num_paid=self.cfg.num_paid,
                    )
                    gs = self.tournament_structure.to_inner_game_string_for_state(
                        blind_level=sampled["blind_level"],
                        stacks=sampled["stacks"],
                        dealer_seat=sampled["dealer_seat"],
                    )
                    game = pyspiel.load_game(gs)
                    state = game.new_initial_state()
                    ctx = CFR6MaxContext(
                        policy_nets=self.policy_nets,
                        encoder=self.encoder,
                        starting_stacks=list(sampled["stacks"]),
                        payouts=self.payouts,
                        iteration=it,
                        max_depth=self.cfg.max_traversal_depth,
                        num_paid=self.cfg.num_paid,
                    )
                    opp_override = self._maybe_sample_league_opponent()
                    traverse_6max(
                        state,
                        traversing_player=traverser,
                        ctx=ctx,
                        rng=self.rng,
                        opponent_policy_override=opp_override,
                    )

            # Train the traverser's advantage net.
            adv_loss = self._train_advantage_net(traverser)

            elapsed = time.time() - t_it
            metrics["iter"].append(it)
            metrics["time"].append(elapsed)
            metrics["traverser"].append(traverser)
            metrics["adv_loss"].append(adv_loss)
            for s in range(NUM_SEATS_6MAX):
                metrics[f"buf_{s}"].append(len(self.policy_nets.buffer_for(s)))

            self.log(
                f"iter {it:>4}/{self.cfg.n_iterations}  "
                f"trav={traverser}  "
                f"adv={'nan' if math.isnan(adv_loss) else f'{adv_loss:.4f}':>8}  "
                f"bufs=({', '.join(str(len(self.policy_nets.buffer_for(s))) for s in range(NUM_SEATS_6MAX))})  "
                f"{elapsed:.1f}s"
            )

            if checkpoint_dir is not None and (
                it % checkpoint_every == 0 or it == self.cfg.n_iterations
            ):
                ckpt_path = checkpoint_dir / f"ckpt_iter_{it:04d}.pt"
                self.save_checkpoint(ckpt_path, slim=True)
                self.log(f"  saved checkpoint: {ckpt_path}")

        total = time.time() - t_start
        self.log(f"=== total: {total/60:.1f} min ===")
        return metrics

    # ---- Checkpoint ----

    def save_checkpoint(self, path: str | Path, slim: bool = False) -> None:
        """Persist solver state for bit-identical resumable training.

        Saves:
          - PlayerNetworks6Max.state_dict() (all 6 nets + optimizers)
          - per-seat buffer state (features, targets, masks, iters, n_seen, rng)
          - current iteration
          - Python RNG state + torch RNG state
          - config dict (for verification on resume)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "iteration": self.iteration,
            "policy_nets": self.policy_nets.state_dict(),
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "config_dict": asdict(self.cfg),
            "slim": slim,
        }
        if not slim:
            ckpt["buffers"] = [
                {
                    "features": list(b.features),
                    "targets": list(b.targets),
                    "legal_masks": list(b.legal_masks),
                    "iters": list(b.iters),
                    "n_seen": b.n_seen,
                    "rng_state": b.rng.getstate(),
                }
                for b in self.policy_nets.buffers
            ]
        torch.save(ckpt, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore from a checkpoint produced by save_checkpoint.

        Bit-identical resume: parameters, optimizer states, and buffer
        contents are restored to their saved values; subsequent traversals
        with the same RNG state produce the same trajectories.
        """
        ckpt = torch.load(str(path), weights_only=False, map_location=self.device)
        self.iteration = ckpt["iteration"]
        self.policy_nets.load_state_dict(ckpt["policy_nets"])

        if "buffers" in ckpt and not ckpt.get("slim", False):
            if len(ckpt["buffers"]) != NUM_SEATS_6MAX:
                raise ValueError(
                    f"checkpoint has {len(ckpt['buffers'])} buffers; expected {NUM_SEATS_6MAX}"
                )
            for i, b_data in enumerate(ckpt["buffers"]):
                buf = self.policy_nets.buffer_for(i)
                buf.features = list(b_data["features"])
                buf.targets = list(b_data["targets"])
                buf.legal_masks = list(b_data["legal_masks"])
                buf.iters = list(b_data["iters"])
                buf.n_seen = b_data["n_seen"]
                buf.rng.setstate(b_data["rng_state"])

        self.rng.setstate(ckpt["rng_state"])
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
        except (TypeError, RuntimeError) as e:
            # Cross-version / cross-device torch RNG formats may differ.
            # Resuming without RNG continuity still produces a valid
            # (just non-byte-identical) continuation.
            self.log(
                f"Note: skipping torch RNG state restore "
                f"({type(e).__name__}: {e})"
            )
