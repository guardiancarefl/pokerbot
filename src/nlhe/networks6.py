"""6-player advantage networks for Phase 4e.3 (CFR training).

Phase 4e.3a (docs/PHASE4_PLAN.md). The data structure that holds the
six per-player advantage networks + their optimizers + their reservoir
buffers. The CFR training loop (Phase 4e.3b/c) will use this as the
indexed network container.

In external-sampling CFR, each player needs their own regret signal
because their information sets and strategic incentives differ. For
HUNL (Phase 2d) this was 2 networks. For 6-max it's 6.

Architecture per network: MLP same as HUNL solver, output dim = 7
(matching N_DISCRETE_ACTIONS from the action abstraction). Input dim
matches the InfosetEncoder6Max feature_dim (236 by default).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reuse the existing MLP architecture + RM+ helper from the HUNL solver.
from src.nlhe.solver import MLP, ReservoirBuffer, _strategy_from_advantages
from src.nlhe.actions import DiscreteAction


N_DISCRETE_ACTIONS = len(DiscreteAction)
NUM_SEATS_6MAX = 6

# Checkpoint schema_version "v2_with_strategy": adds a single shared strat_net +
# strat_optimizer to the v1 advantage-only container. The strat_buffer CONTENTS
# are NOT in state_dict() — solver6.save_checkpoint serializes them alongside the
# advantage buffers so they respect slim-checkpoint mode (see DECISIONS.md).
# TWO-TIER schema handling (revised in Step E): load_state_dict ACCEPTS both v1
# and v2 dicts (v1 loads advantage nets only, marks the container "v1"); the
# refuse-on-mismatch moved to the deployment boundary — inference_policy uses the
# strategy net only on v2, else falls back to the regret-matched current strategy.
SCHEMA_VERSION = "v2_with_strategy"


@dataclass
class PlayerNetworks6Max:
    """Container for 6 per-seat advantage networks + optimizers + buffers, plus
    one shared strategy network + optimizer + shared strategy buffer (v2 schema,
    see DECISIONS.md). Strategy net learns the position-conditioned average policy
    across all 6 seats from the shared buffer; seat is encoded in the feature
    vector, not in network topology (Pluribus-style sharing).

    Args:
        input_dim: feature dim (matches InfosetEncoder6Max.feature_dim;
            default 236).
        hidden: hidden layer sizes for each MLP (e.g. [256, 256]).
        learning_rate: per-network Adam LR (shared by the strategy net).
        buffer_capacity: reservoir buffer capacity (per-seat adv buffers and
            the single shared strategy buffer).
        rng: random source for the 6 advantage buffers' reservoir sampling
            (one instance shared across all 6 adv buffers).
        device: 'cpu' or 'cuda'.
        strat_rng: INDEPENDENT random source for the strategy buffer. Kept
            separate from `rng` so strategy writes/sampling never perturb the
            advantage-buffer rng stream (preserves advantage-net bit-identity).
            If None, __post_init__ assigns a fresh random.Random(); production
            (solver6) passes random.Random(config.seed + 100) — HUNL's offset.
    """
    input_dim: int = 236
    hidden: list[int] = field(default_factory=lambda: [256, 256])
    learning_rate: float = 1e-3
    buffer_capacity: int = 100_000
    rng: random.Random = field(default_factory=random.Random)
    device: str = "cpu"
    strat_rng: random.Random | None = None

    # Populated in __post_init__.
    nets: list[MLP] = field(default_factory=list, init=False)
    optimizers: list[optim.Optimizer] = field(default_factory=list, init=False)
    buffers: list[ReservoirBuffer] = field(default_factory=list, init=False)
    # Single shared strategy net + optimizer + buffer (v2 schema).
    strat_net: MLP = field(default=None, init=False)
    strat_optimizer: optim.Optimizer = field(default=None, init=False)
    strat_buffer: ReservoirBuffer = field(default=None, init=False)
    # Schema of the most-recently-loaded checkpoint (or SCHEMA_VERSION for a
    # freshly-constructed / freshly-trained container, which is v2 by definition).
    # Two-tier design: load_state_dict ACCEPTS both schemas; inference_policy
    # only USES the strategy net when this is the v2 schema, else falls back to
    # the regret-matched current strategy (advantage net).
    loaded_schema_version: str = field(default=SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        if len(self.nets) != 0:
            # Allow caller to pre-populate (e.g. when loading checkpoints).
            return

        for seat in range(NUM_SEATS_6MAX):
            net = MLP(in_dim=self.input_dim, hidden=self.hidden,
                      out_dim=N_DISCRETE_ACTIONS).to(self.device)
            self.nets.append(net)
            self.optimizers.append(optim.Adam(net.parameters(), lr=self.learning_rate))
            # Each seat gets its own buffer with its own rng — but they share
            # a single rng instance because we want determinism from a single
            # seed. Each call to add() uses the shared rng's state.
            self.buffers.append(ReservoirBuffer(capacity=self.buffer_capacity, rng=self.rng))

        # ---- Shared strategy net + optimizer + buffer (v2 schema) ----
        # ONE net (not per-seat): seat is in the feature vector, so a single net
        # learns the position-conditioned average policy (Pluribus-style sharing).
        # strat_rng is resolved FIRST and is INDEPENDENT of self.rng (the adv-
        # buffer rng), so strategy writes/sampling never advance the advantage
        # rng stream — this is what keeps advantage-net training bit-identical.
        if self.strat_rng is None:
            self.strat_rng = random.Random()
        self.strat_net = MLP(in_dim=self.input_dim, hidden=self.hidden,
                             out_dim=N_DISCRETE_ACTIONS).to(self.device)
        self.strat_optimizer = optim.Adam(self.strat_net.parameters(), lr=self.learning_rate)
        # Mirrors the adv-buffer pattern: pass the rng INSTANCE (self.strat_rng),
        # not a seed int. Instantiated after strat_rng is resolved above.
        self.strat_buffer = ReservoirBuffer(capacity=self.buffer_capacity, rng=self.strat_rng)

    # ---- Access by seat ----

    def net_for(self, seat: int) -> MLP:
        self._check_seat(seat)
        return self.nets[seat]

    def optimizer_for(self, seat: int) -> optim.Optimizer:
        self._check_seat(seat)
        return self.optimizers[seat]

    def buffer_for(self, seat: int) -> ReservoirBuffer:
        self._check_seat(seat)
        return self.buffers[seat]

    def _check_seat(self, seat: int) -> None:
        if not (0 <= seat < NUM_SEATS_6MAX):
            raise IndexError(f"seat {seat} out of range [0, {NUM_SEATS_6MAX})")

    # ---- Device movement ----

    def to(self, device: str) -> "PlayerNetworks6Max":
        """Move all 6 nets to the given device. Optimizer state moves with
        the nets (Adam state is tied to parameter tensors).
        """
        self.device = device
        for net in self.nets:
            net.to(device)
        if self.strat_net is not None:
            self.strat_net.to(device)
        return self

    # ---- Forward pass convenience ----

    def predict_advantages(self, seat: int, features: np.ndarray) -> np.ndarray:
        """Run a forward pass for the given seat. Returns numpy array of
        shape (n_actions,).
        """
        self._check_seat(seat)
        x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.nets[seat](x)
        return out.squeeze(0).cpu().numpy()

    def inference_policy(
        self, seat: int, features: np.ndarray, legal_mask: np.ndarray
    ) -> np.ndarray:
        """Deployment-time policy distribution over the discrete actions.

        Two-tier dispatch on the loaded checkpoint schema:
          - v2 ("v2_with_strategy"): the shared strategy net's masked softmax —
            the deployment-quality average policy.
          - v1: the regret-matched current strategy from the seat's advantage
            net (_strategy_from_advantages) — the backward-compatible legacy path.

        Returns a (N_DISCRETE_ACTIONS,) distribution, zero on illegal actions and
        summing to 1 on the legal subset. RNG-NEUTRAL: no draws here — the caller
        samples/argmaxes from the returned distribution.
        """
        self._check_seat(seat)
        if self.loaded_schema_version == SCHEMA_VERSION:
            self.strat_net.eval()
            x = torch.from_numpy(
                np.asarray(features, dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.strat_net(x).squeeze(0).cpu().numpy()
            # Masked softmax (softmax -> mask -> renormalize), mirroring
            # _train_strategy_net and HUNL policy_adapter.py, with a
            # uniform-over-legal fallback if the masked mass underflows.
            logits = logits - logits.max()
            exp_l = np.exp(logits) * legal_mask
            denom = float(exp_l.sum())
            if denom > 0.0:
                return exp_l / denom
            return legal_mask / float(legal_mask.sum())
        # v1: regret-matched current strategy from the advantage net (legacy).
        adv = self.predict_advantages(seat, features)
        return _strategy_from_advantages(adv, legal_mask)

    # ---- Checkpoint state ----

    def state_dict(self) -> dict:
        """Snapshot the 6 advantage nets + the shared strategy net (v2 schema).

        Carries net + optimizer parameters and config scalars. The strategy
        BUFFER contents are intentionally NOT serialized here — solver6.
        save_checkpoint persists them alongside the advantage buffers so they
        honor slim-checkpoint mode (C1 design contract, DECISIONS.md).
        """
        return {
            "schema_version": SCHEMA_VERSION,
            "nets": [n.state_dict() for n in self.nets],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "strat_net": self.strat_net.state_dict(),
            "strat_optimizer": self.strat_optimizer.state_dict(),
            "input_dim": self.input_dim,
            "hidden": list(self.hidden),
            "learning_rate": self.learning_rate,
            "buffer_capacity": self.buffer_capacity,
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore from a state_dict produced by self.state_dict().

        TWO-TIER schema handling (a deliberate revision of the original
        refuse-on-load behavior — see DECISIONS.md):

          * LOAD accepts BOTH schemas. A v2 dict ("v2_with_strategy") restores
            the advantage nets AND the shared strategy net/optimizer. A v1 dict
            (no schema_version) restores ONLY the advantage nets/optimizers and
            leaves the strategy net at fresh init — so the 36 existing v1
            checkpoints stay loadable for advantage-net-only paths (eval,
            sub-step-6 reproduction).
          * INFERENCE refuses to USE the strategy net on a v1 load. The detected
            schema is recorded in self.loaded_schema_version, and inference_policy
            dispatches on it: v2 -> strategy net; v1 -> regret-matched current
            (advantage net). The "refuse-on-mismatch" moves from the load
            boundary to the deployment boundary.
        """
        if len(sd["nets"]) != NUM_SEATS_6MAX:
            raise ValueError(f"state_dict has {len(sd['nets'])} nets, expected {NUM_SEATS_6MAX}")
        for i, net_sd in enumerate(sd["nets"]):
            self.nets[i].load_state_dict(net_sd)
        for i, opt_sd in enumerate(sd["optimizers"]):
            self.optimizers[i].load_state_dict(opt_sd)

        if sd.get("schema_version") == SCHEMA_VERSION:
            self.strat_net.load_state_dict(sd["strat_net"])
            self.strat_optimizer.load_state_dict(sd["strat_optimizer"])
            self.loaded_schema_version = SCHEMA_VERSION
        else:
            # v1 (pre-strategy-net) checkpoint: advantage nets only. Strategy net
            # stays at fresh init; inference_policy will fall back to the
            # regret-matched current strategy for this container.
            self.loaded_schema_version = "v1"
