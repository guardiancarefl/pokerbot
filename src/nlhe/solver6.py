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

  2. Strategy net trained. PlayerNetworks6Max carries a single shared strategy
     net + buffer (v2 schema). cfr6.traverse_6max writes the acting seat's
     current regret-matched policy to the shared buffer at non-traverser opp
     nodes; _train_strategy_net trains the net on it each iteration (KL,
     _dcfr_weights). The strategy net is the deployment-quality average policy;
     consumers select it over the regret-matched current strategy on v2
     checkpoints (Step E).

  3. DCFR weighting wired in. _dcfr_weights (cfr_variant vanilla|linear|
     discounted) applies to both advantage-net (line ~393) and strategy-
     net (line ~444) training. The dcfr-overnight-3000 baseline used
     cfr_variant="linear", dcfr_exponent=1.0.

  4. Archetype mix wired in. src/nlhe/archetype6.py wraps HUNL's
     archetype_policy as an ArchetypePolicy + ArchetypePool, sampled by
     the three-way combined override-slot roll in
     _maybe_sample_league_opponent. Controlled by archetype_mix +
     archetype_calibration_path + archetype_profiles. Off (mix=0.0) by
     default → bit-identical to pre-archetype behavior.

  5. Tournament-aware stack sampling wired in. When tournament_structure_
     path is set, stack_sampler.sample_starting_state supplies per-traversal
     stacks/blind-level/dealer (lines ~518-551) and ICM uses the sampled
     stacks. The uniform [cfg.starting_stack] * 6 path remains as the
     fallback when tournament_structure is unset (e.g. six_max_smoke for
     development).

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

# Pin BLAS to single-threaded across the codebase. Determinism under mp.fork() is a
# project-wide property: forked workers inherit MKL/OpenMP thread-pool state from the
# parent and can deadlock on the first BLAS call if multi-threaded BLAS was initialized
# in the parent. The orchestrator gets parallelism from processes, not intra-op threads.
torch.set_num_threads(1)

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


# ===== Helpers =====


def _fmt_hms(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS (Step 7 dashboard ETA/elapsed)."""
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


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


# Knuth's odd-constant mixing salt — independent stream offset for override sampling.
# Combined with the (seed, iter, traversal_id) per-traversal fork formula, this gives a
# second deterministic stream statistically independent of the traversal rng, used by
# _maybe_sample_league_opponent so override band/pool sampling doesn't cascade into the
# traversal stream (which would break parallel bit-identity at mix > 0).
OVERRIDE_SALT = 0x9E3779B97F4A7C15

# MurmurHash2 64-bit mixer constant — third independent salted stream for per-traversal
# stack sampling in tournament mode. Workers and the sequential train() loop both derive
# rng_stack_t from (seed, it, t) + STACK_SAMPLE_SALT, so sample_starting_state's variable
# rng-draw count never perturbs the traversal or override streams (preserves parallel
# bit-identity at tournament_structure_path != None).
STACK_SAMPLE_SALT = 0xC6A4A7935BD1E995


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

    # Archetype-opponent mix (Pillar 3 — style diversity). Probability per
    # traversal that the opp override comes from the archetype pool instead
    # of the league pool. PLACEHOLDER in Phase 5-A: the archetype band returns
    # None until the archetype Policy is ported in Phase 5-B. Combined with
    # league_mix via a single three-way roll in _maybe_sample_league_opponent:
    # archetype_mix + league_mix must be <= 1.0; the remainder is self-play.
    # Defaults to 0.0 → bit-identical to pre-Phase-5-A behavior.
    archetype_mix: float = 0.0
    # Path to the 6-max EquityCalibration JSON (Phase 5-pre artifact). Required
    # when archetype_mix > 0 (validated in the solver __init__, mirroring the
    # league_mix>0 requires registry pattern). Default None → no archetype pool.
    archetype_calibration_path: Optional[str] = None
    # Subset of archetype profiles to sample from. None → all five
    # (NIT, TAG, LAG, STATION, MANIAC). Validated as a subset in __post_init__.
    archetype_profiles: Optional[list] = None

    # --- Step 7 dashboard (Pieces A + B). All default OFF → bit-identical to
    # pre-dashboard behavior. Pure observability; no training-trajectory effect.
    enhanced_logging: bool = False          # Piece A: ETA + rolling avgs + override-mix
    mini_eval_enabled: bool = False         # Piece B: periodic head-to-head vs anchors
    mini_eval_every: int = 200              # mini-eval cadence (iters)
    mini_eval_n_hands: int = 200            # hands per anchor per snapshot
    mini_eval_anchors: Optional[list] = None         # "name=path" specs (or bare paths) to anchor .pt
    mini_eval_shanky_rotation: Optional[list] = None  # Shanky profile specs, cycled one per snapshot

    seed: int = 2026

    def __post_init__(self):
        # Config-vs-config invariants (first validation hook on this dataclass).
        # Individual mix bounds keep the dataclass self-consistent; the sum
        # bound is the new cross-field constraint introduced by the combined
        # archetype+league override slot (Phase 5-A). The override probability
        # mass cannot exceed 1.0 — the remainder is self-play.
        if not 0.0 <= self.archetype_mix <= 1.0:
            raise ValueError(
                f"archetype_mix must be in [0.0, 1.0], got {self.archetype_mix}"
            )
        if not 0.0 <= self.league_mix <= 1.0:
            raise ValueError(
                f"league_mix must be in [0.0, 1.0], got {self.league_mix}"
            )
        if self.archetype_mix + self.league_mix > 1.0:
            raise ValueError(
                "archetype_mix + league_mix must be <= 1.0 (the remainder is "
                f"self-play); got archetype_mix={self.archetype_mix}, "
                f"league_mix={self.league_mix}, "
                f"sum={self.archetype_mix + self.league_mix}"
            )
        # archetype_profiles, when given, must be a subset of the named
        # archetypes. Config-only check (no runtime objects) → __post_init__.
        if self.archetype_profiles is not None:
            from src.nlhe.archetype6 import VALID_ARCHETYPE_NAMES
            bad = [p for p in self.archetype_profiles
                   if p not in VALID_ARCHETYPE_NAMES]
            if bad:
                raise ValueError(
                    f"archetype_profiles {bad} not in valid archetype names "
                    f"{VALID_ARCHETYPE_NAMES}"
                )
        # Step 7 dashboard config-only invariants. Anchor-file existence is a
        # runtime check (solver __init__), mirroring league_mix>0 requires path.
        if self.mini_eval_enabled:
            if not self.mini_eval_anchors and not self.mini_eval_shanky_rotation:
                raise ValueError(
                    "mini_eval_enabled=True requires at least one of "
                    "mini_eval_anchors or mini_eval_shanky_rotation to be "
                    "non-empty."
                )
            if self.mini_eval_every < 1:
                raise ValueError(
                    f"mini_eval_every must be >= 1, got {self.mini_eval_every}"
                )
            if self.mini_eval_n_hands < 1:
                raise ValueError(
                    f"mini_eval_n_hands must be >= 1, got {self.mini_eval_n_hands}"
                )


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
        # max_bucket_dim auto-derived from the abstraction so the feature/net
        # dims match whatever k the abstraction was built with. Prevents silent
        # one-hot truncation if a higher-k abstraction (e.g. k=500) is loaded
        # without explicitly updating the encoder dimension.
        max_bucket_dim = max(sa.k for sa in abstraction.streets.values())
        self.encoder = InfosetEncoder6Max(
            abstraction=abstraction,
            starting_stack=config.starting_stack,
            max_bucket_dim=max_bucket_dim,
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
            strat_rng=random.Random(config.seed + 100),
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

        # Archetype play (Pillar 3 — style diversity). Symmetric with league:
        # build the pool when a calibration path is set; archetype_mix > 0 with
        # no path is a configuration error. archetype_mix == 0 with a path is
        # allowed (pool built but never sampled) — useful for the F2 gate.
        self.archetype_pool = None
        if config.archetype_mix > 0.0 and not config.archetype_calibration_path:
            raise ValueError(
                "archetype_mix > 0 requires archetype_calibration_path to be "
                f"set; got archetype_mix={config.archetype_mix}, "
                f"archetype_calibration_path={config.archetype_calibration_path!r}"
            )
        if config.archetype_calibration_path:
            from src.nlhe.archetype6 import ArchetypePool
            self.archetype_pool = ArchetypePool(
                calibration_path=config.archetype_calibration_path,
                abstraction=abstraction,
                profile_names=config.archetype_profiles,
                bucket_runouts=config.bucket_runouts,
            )
            self.log(
                f"  archetype pool: {len(self.archetype_pool)} profiles, "
                f"mix={config.archetype_mix:.3f}"
            )

        # Step 7 dashboard state. Override-mix counters (Piece A) — reset each
        # iter in train(); incremented in _maybe_sample_league_opponent. Pure
        # observability (no rng draw → bit-identity-inert).
        self._override_counts = {"archetype": 0, "league": 0, "self_play": 0}
        # Mini-eval (Piece B): per-snapshot result history + anchor policy cache.
        self._mini_eval_history: list = []
        self._mini_eval_anchor_cache: dict = {}
        if config.mini_eval_enabled:
            import os as _os
            specs = list(config.mini_eval_anchors or []) + list(
                config.mini_eval_shanky_rotation or [])
            for spec in specs:
                path = spec.split("=", 1)[1] if "=" in spec else spec
                if not _os.path.exists(path):
                    raise ValueError(f"mini_eval anchor/profile not found: {path}")
            n_anchors = len(config.mini_eval_anchors or [])
            n_shanky = len(config.mini_eval_shanky_rotation or [])
            self.log(
                f"  mini-eval enabled: {n_anchors} anchors"
                f"{f' + {n_shanky} shanky' if n_shanky else ''}"
                f", every {config.mini_eval_every} iters, "
                f"{config.mini_eval_n_hands} hands/anchor"
            )

        self.iteration = 0

        self.log(
            f"DeepCFR6MaxSolver  device={self.device}  "
            f"feature_dim={self.encoder.feature_dim}  "
            f"payout_mode={config.payout_mode}  payouts={self.payouts}"
        )

    # ---- Network training ----

    def _maybe_sample_league_opponent(
        self,
        rng: Optional[random.Random] = None,
        count_only: bool = False,
    ):
        """Sample this traversal's opponent override, or None (self-play).

        Combined three-way override-slot sampling (Phase 5-A). One uniform
        roll r on the supplied rng (or self.rng if not provided) partitions
        the unit interval into three bands:

            archetype  [0, archetype_mix)
            league     [archetype_mix, archetype_mix + league_mix)
            self-play  [archetype_mix + league_mix, 1)   → None

        Returns:
            An ArchetypePolicy (archetype band, Phase 5-B), a Policy
            (CheckpointPolicy / ShankyProfilePolicy, league band per LeaguePool),
            or None (self-play band). When count_only=True, returns None in
            every band (the caller only wants the counter increment and the
            deterministic rng draw, not a Policy object).

        Both pools expose sample_opponent(rng) with NO internal mix gate — this
        roll owns the mix decision. Archetype opponents reach training via the
        cfr6 NON-traverser short-circuit, so their decisions are never written
        to the strategy buffer (DECISIONS.md:216).

        Bit-identity at the default: when no override source is active (no
        archetype mass AND no usable league pool/mix), we short-circuit with
        NO rng draw — exactly matching the pre-Phase-5-A behavior. And at
        archetype_mix=0.0 the archetype band [0, 0) is empty, so the league
        band collapses to [0, league_mix) — the original single-gate
        condition, consuming the identical rng draw. The override remains
        per-traversal-shared across all opponent seats (unchanged semantics).

        Phase 1 override-fork: callers that need parallel bit-identity at
        mix>0 supply an independent per-traversal rng (derived via the
        OVERRIDE_SALT formula in train()), so override-band sampling never
        perturbs the traversal stream. When rng is None, falls back to
        self.rng to preserve callers (e.g. mid-migration paths) that haven't
        adopted the explicit fork yet.

        Phase 2 count_only: at mix>0 the parallel orchestrator's merge phase
        needs to advance the override counter (and the rng draw, if rng is
        self.rng) without paying the cost of sampling a Policy from the pool
        — workers do the actual sampling on their side using the same
        deterministic rng_override_t. count_only=True produces the same
        rng draws and the same _count_override increments as count_only=False
        but returns None instead of constructing a Policy.
        """
        rng = rng or self.rng
        archetype_active = self.archetype_pool is not None and self.cfg.archetype_mix > 0.0
        league_active = self.league_pool is not None and self.cfg.league_mix > 0.0
        if not archetype_active and not league_active:
            # No override source → self-play, no rng draw (preserves
            # bit-identity with the pre-Phase-5-A short-circuit).
            self._count_override("self_play")
            return None

        r = rng.random()
        if r < self.cfg.archetype_mix:
            # Archetype band (Phase 5-B): sample a style profile as an override
            # Policy. The pool has no internal mix gate — the roll above already
            # placed us in this band.
            if self.archetype_pool is None:
                self._count_override("self_play")
                return None
            self._count_override("archetype")
            if count_only:
                return None
            return self.archetype_pool.sample_opponent(rng)
        if r < self.cfg.archetype_mix + self.cfg.league_mix:
            # League band.
            if self.league_pool is None:
                self._count_override("self_play")
                return None
            self._count_override("league")
            if count_only:
                return None
            return self.league_pool.sample_opponent(rng)
        # Self-play band.
        self._count_override("self_play")
        return None

    def _count_override(self, band: str) -> None:
        """Increment the per-iter override-mix counter (Piece A). Defensive:
        no-op when the counter is absent (e.g. duck-typed test fakes). Never
        draws rng — bit-identity-inert."""
        counts = getattr(self, "_override_counts", None)
        if counts is not None:
            counts[band] = counts.get(band, 0) + 1

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

    def _train_strategy_net(self) -> float:
        """One iteration of training the single shared strategy net.

        Trains on the shared strategy buffer (samples written at non-traverser
        opp nodes in cfr6.traverse_6max). Returns mean KL loss across
        `train_steps_per_iter` batches, or NaN if the buffer is smaller than
        batch_size (matches the advantage-net / HUNL convention).

        Mirrors HUNL src/nlhe/solver.py::_train_strategy_net: KL between the
        buffered regret-matched policy target and the net's masked softmax
        (softmax-then-mask-and-renormalize), per-sample then DCFR-weighted via
        the SAME _dcfr_weights as the advantage side (vanilla -> None ->
        unweighted mean). ONE shared net, so no seat argument — position is in
        the feature vector. The buffer samples via strat_rng, independent of the
        traversal rng (Step B C2), so this never perturbs the advantage path.
        """
        buf = self.policy_nets.strat_buffer
        if len(buf) < self.cfg.batch_size:
            return float("nan")

        net = self.policy_nets.strat_net
        opt = self.policy_nets.strat_optimizer
        net.train()

        total_loss = 0.0
        for _ in range(self.cfg.train_steps_per_iter):
            feats, targets, masks, iters = buf.sample_batch(self.cfg.batch_size)
            feats = feats.to(self.device)
            targets = targets.to(self.device)
            masks = masks.to(self.device)
            iters = iters.to(self.device)

            # Masked softmax over actions; KL(target || probs). Softmax-then-
            # mask-and-renormalize, matching HUNL solver.py exactly.
            logits = net(feats)
            logits = logits - logits.max(dim=1, keepdim=True).values  # numerical stability
            exp_l = torch.exp(logits) * masks
            denom = exp_l.sum(dim=1, keepdim=True).clamp(min=1e-8)
            probs = exp_l / denom
            per_sample = -(targets * torch.log(probs + 1e-8) * masks).sum(dim=1)
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
            adv_loss (for the traversed seat), strat_loss (shared strategy
            net), per-seat advantage buffer sizes, and strat_buf (shared
            strategy buffer size).
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        start_iter = self.iteration + 1
        metrics: dict = {
            "iter": [], "time": [], "traverser": [], "adv_loss": [],
            "strat_loss": [], "strat_buf": [], "mini_eval": [],
        }
        for s in range(NUM_SEATS_6MAX):
            metrics[f"buf_{s}"] = []

        t_start = time.time()
        for it in range(start_iter, self.cfg.n_iterations + 1):
            self.iteration = it
            traverser = (it - 1) % NUM_SEATS_6MAX
            t_it = time.time()
            # Reset the per-iter override-mix counter (Piece A). No rng draw.
            self._override_counts = {"archetype": 0, "league": 0, "self_play": 0}

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
                for t in range(self.cfg.traversals_per_iter):
                    rng_t = random.Random(
                        (self.cfg.seed * 1_000_003 + it * 9_973 + t)
                        & 0x7FFFFFFFFFFFFFFF
                    )
                    rng_override_t = random.Random(
                        (self.cfg.seed * 1_000_003 + it * 9_973 + t + OVERRIDE_SALT)
                        & 0x7FFFFFFFFFFFFFFF
                    )
                    state = self.game.new_initial_state()
                    opp_override = self._maybe_sample_league_opponent(rng=rng_override_t)
                    traverse_6max(
                        state,
                        traversing_player=traverser,
                        ctx=ctx,
                        rng=rng_t,
                        opponent_policy_override=opp_override,
                    )
            else:
                # Tournament-aware mode: sample state + build game per trajectory.
                from src.nlhe.stack_sampler import sample_starting_state
                import pyspiel
                for t in range(self.cfg.traversals_per_iter):
                    rng_t = random.Random(
                        (self.cfg.seed * 1_000_003 + it * 9_973 + t)
                        & 0x7FFFFFFFFFFFFFFF
                    )
                    rng_override_t = random.Random(
                        (self.cfg.seed * 1_000_003 + it * 9_973 + t + OVERRIDE_SALT)
                        & 0x7FFFFFFFFFFFFFFF
                    )
                    rng_stack_t = random.Random(
                        (self.cfg.seed * 1_000_003 + it * 9_973 + t + STACK_SAMPLE_SALT)
                        & 0x7FFFFFFFFFFFFFFF
                    )
                    sampled = sample_starting_state(
                        self.tournament_structure,
                        rng_stack_t,
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
                        dealer_seat=sampled["dealer_seat"],
                    )
                    opp_override = self._maybe_sample_league_opponent(rng=rng_override_t)
                    traverse_6max(
                        state,
                        traversing_player=traverser,
                        ctx=ctx,
                        rng=rng_t,
                        opponent_policy_override=opp_override,
                    )

            # Train the traverser's advantage net.
            adv_loss = self._train_advantage_net(traverser)

            # Train the single shared strategy net on the shared strategy
            # buffer (samples written at opp nodes during the traversals above).
            # Mirrors HUNL ordering: strategy training follows advantage training.
            strat_loss = self._train_strategy_net()

            elapsed = time.time() - t_it
            metrics["iter"].append(it)
            metrics["time"].append(elapsed)
            metrics["traverser"].append(traverser)
            metrics["adv_loss"].append(adv_loss)
            metrics["strat_loss"].append(strat_loss)
            metrics["strat_buf"].append(len(self.policy_nets.strat_buffer))
            for s in range(NUM_SEATS_6MAX):
                metrics[f"buf_{s}"].append(len(self.policy_nets.buffer_for(s)))

            if self.cfg.enhanced_logging:
                self._log_enhanced(it, traverser, adv_loss, strat_loss,
                                   elapsed, t_start, metrics)
            else:
                # Default path — BYTE-IDENTICAL to pre-dashboard behavior.
                self.log(
                    f"iter {it:>4}/{self.cfg.n_iterations}  "
                    f"trav={traverser}  "
                    f"adv={'nan' if math.isnan(adv_loss) else f'{adv_loss:.4f}':>8}  "
                    f"strat={'nan' if math.isnan(strat_loss) else f'{strat_loss:.4f}':>8}  "
                    f"bufs=({', '.join(str(len(self.policy_nets.buffer_for(s))) for s in range(NUM_SEATS_6MAX))})  "
                    f"sbuf={len(self.policy_nets.strat_buffer)}  "
                    f"{elapsed:.1f}s"
                )

            # Checkpoint BEFORE mini-eval so the most recent checkpoint exists
            # on disk by the time mini-eval looks up the "self from prev cycle"
            # anchor (the file is ckpt_iter_{it-mini_eval_every:04d}.pt, which
            # was written `mini_eval_every` iters ago). Order swap from the
            # original (eval then save) is mini-eval-only: at mix=0 + mini_eval
            # disabled, the trajectory is bit-identical to pre-swap behavior
            # because nothing observes the ordering except the self-anchor
            # lookup inside _maybe_run_mini_eval.
            if checkpoint_dir is not None and (
                it % checkpoint_every == 0 or it == self.cfg.n_iterations
            ):
                ckpt_path = checkpoint_dir / f"ckpt_iter_{it:04d}.pt"
                self.save_checkpoint(ckpt_path, slim=True)
                self.log(f"  saved checkpoint: {ckpt_path}")

            # Mini-eval (Piece B): periodic strength heartbeat. Uses an isolated
            # eval seed (cfg.seed+200+it) — never touches self.rng.
            if self.cfg.mini_eval_enabled and it % self.cfg.mini_eval_every == 0:
                self._maybe_run_mini_eval(it, metrics, checkpoint_dir)

        total = time.time() - t_start
        self.log(f"=== total: {total/60:.1f} min ===")
        return metrics

    # ---- Step 7 dashboard (Pieces A + B) ----

    def _log_enhanced(self, it, traverser, adv_loss, strat_loss, elapsed,
                      t_start, metrics) -> None:
        """Piece A: enhanced per-iter log — ETA, rolling avg10, override-mix,
        per-iter wall. Pure observability; no rng draw, no training change."""
        def avg10(key):
            xs = [x for x in metrics[key][-10:] if not (isinstance(x, float) and math.isnan(x))]
            return sum(xs) / len(xs) if xs else float("nan")

        def fnum(x):
            return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"

        wall_elapsed = time.time() - t_start
        wall_avg10 = avg10("time")
        remaining = self.cfg.n_iterations - it
        eta = remaining * wall_avg10 if wall_avg10 == wall_avg10 else 0.0  # nan-guard

        c = self._override_counts
        tot = max(sum(c.values()), 1)
        mix = (f"arch={100*c['archetype']/tot:.1f}% "
               f"league={100*c['league']/tot:.1f}% "
               f"self={100*c['self_play']/tot:.1f}%")

        self.log(
            f"iter {it:>4}/{self.cfg.n_iterations}  "
            f"[elapsed {_fmt_hms(wall_elapsed)}, ETA {_fmt_hms(eta)}]\n"
            f"    trav={traverser}  "
            f"adv-loss {fnum(adv_loss)} (avg10 {fnum(avg10('adv_loss'))})  "
            f"strat-loss {fnum(strat_loss)} (avg10 {fnum(avg10('strat_loss'))})\n"
            f"    adv-bufs ({', '.join(str(len(self.policy_nets.buffer_for(s))) for s in range(NUM_SEATS_6MAX))})  "
            f"strat-buf {len(self.policy_nets.strat_buffer)}\n"
            f"    override-mix: {mix} (this iter)  "
            f"iter-wall {elapsed:.1f}s (avg10 {wall_avg10:.1f}s)"
        )

    def _maybe_run_mini_eval(self, it, metrics, checkpoint_dir=None) -> None:
        """Piece B: run the periodic mini-eval and log + record results.

        Two output flavors per cycle:
        - lift.log (clean ASCII single-line per anchor) — for grep/post-
          processing/test fixtures. Format asserted by
          test_dashboard_self_anchor.py.
        - lift.log.ansi (colored boxed multi-line block) — for live
          operator viewing via `tail -f`. Also goes to stdout via self.log
          so `tmux attach` shows the colored version.

        Self-anchor (frozen self from one mini_eval_every ago) is added when
        checkpoint_dir is set and the prior checkpoint exists. The first eval
        cycle (no prior checkpoint) emits a placeholder line for log continuity.
        """
        from src.nlhe import mini_eval

        # Locate the "self from prev cycle" checkpoint, if any.
        self_anchor_path = None
        self_anchor_label = None
        prev_iter = it - self.cfg.mini_eval_every
        if checkpoint_dir is not None and prev_iter >= 1:
            candidate = Path(checkpoint_dir) / f"ckpt_iter_{prev_iter:04d}.pt"
            if candidate.exists():
                self_anchor_path = str(candidate)
                self_anchor_label = f"self_iter_{prev_iter:04d}"

        t0 = time.time()
        results = mini_eval.run_mini_eval(
            self, it,
            self_anchor_path=self_anchor_path,
            self_anchor_label=self_anchor_label,
        )
        dt = time.time() - t0

        # When the self-anchor was expected but couldn't be loaded (file
        # missing, first-cycle case), pass a placeholder label to the
        # formatter so the operator still gets one log record per cycle
        # for `grep lift_vs_self` continuity.
        placeholder_label = None
        if self_anchor_label is None and prev_iter >= 1:
            placeholder_label = f"self_iter_{prev_iter:04d}"
        elif self_anchor_label is None and prev_iter < 1:
            placeholder_label = "self_iter_0000"

        n_hands = self.cfg.mini_eval_n_hands
        structure = self.tournament_structure
        prev_results = (
            self._mini_eval_history[-1] if self._mini_eval_history else None
        )

        clean_text = mini_eval.format_eval_block(
            iter_num=it, wall_s=dt, n_hands=n_hands,
            results=results, prev_results=prev_results,
            structure=structure, colored=False,
            placeholder_self_label=placeholder_label,
        )
        colored_text = mini_eval.format_eval_block(
            iter_num=it, wall_s=dt, n_hands=n_hands,
            results=results, prev_results=prev_results,
            structure=structure, colored=True,
            placeholder_self_label=placeholder_label,
        )

        # stdout gets the colored version (operator tmux attach view).
        for line in colored_text.splitlines():
            self.log(line)
        if checkpoint_dir is not None:
            run_dir = Path(checkpoint_dir).parent
            with open(run_dir / "lift.log", "a") as f:
                f.write(clean_text + "\n")
            with open(run_dir / "lift.log.ansi", "a") as f:
                f.write(colored_text + "\n")

        record = {"iter": it, "wall_s": dt, "results": results}
        self._mini_eval_history.append(results)
        metrics["mini_eval"].append(record)

    # ---- Checkpoint ----

    def save_checkpoint(self, path: str | Path, slim: bool = False) -> None:
        """Persist solver state for bit-identical resumable training.

        Saves:
          - PlayerNetworks6Max.state_dict() (6 advantage nets + the shared
            strategy net + optimizers; v2 schema)
          - per-seat advantage buffer state AND the shared strategy buffer
            state (features, targets, masks, iters, n_seen, rng) — non-slim only
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
            # Shared strategy buffer (C1 contract: persisted here, alongside the
            # advantage buffers, so it respects slim mode). Same dict shape.
            sb = self.policy_nets.strat_buffer
            ckpt["strat_buffer"] = {
                "features": list(sb.features),
                "targets": list(sb.targets),
                "legal_masks": list(sb.legal_masks),
                "iters": list(sb.iters),
                "n_seen": sb.n_seen,
                "rng_state": sb.rng.getstate(),
            }
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

        # Shared strategy buffer (C1). On a slim checkpoint there are no buffer
        # contents at all (adv or strat) — leave it empty, no warning (slim
        # contract). On a non-slim checkpoint that unexpectedly lacks it, warn.
        if not ckpt.get("slim", False):
            if "strat_buffer" in ckpt:
                sb_data = ckpt["strat_buffer"]
                sb = self.policy_nets.strat_buffer
                sb.features = list(sb_data["features"])
                sb.targets = list(sb_data["targets"])
                sb.legal_masks = list(sb_data["legal_masks"])
                sb.iters = list(sb_data["iters"])
                sb.n_seen = sb_data["n_seen"]
                sb.rng.setstate(sb_data["rng_state"])
            else:
                self.log(
                    "Note: v2 checkpoint loaded with no strat_buffer contents — "
                    "strategy net will train from an empty buffer."
                )

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
