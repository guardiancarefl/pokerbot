"""Custom Deep CFR (External Sampling) solver for HUNL with card and action abstraction.

This is a hand-rolled CFR loop, intentionally not wrapping OpenSpiel's
DeepCFRSolver, because we need to:
  - Encode infosets via our InfosetEncoder (abstracted card buckets, not raw cards)
  - Restrict action space to our 7 DiscreteAction set, not OpenSpiel's ~20000-action raw space
  - Apply our own per-iteration logging and resumable checkpointing

Algorithm: External Sampling Deep CFR (Brown et al. 2019).
  - At each decision node for the traverser, compute counterfactual regrets
    over ALL legal discrete actions by recursive evaluation.
  - At each decision node for the opponent, sample ONE action from current strategy.
  - Two networks per player: advantage (regret) net and strategy (policy) net.
  - Reservoir buffers preserve uniform samples over training history.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.nlhe.abstraction import Abstraction
from src.nlhe.actions import (
    DiscreteAction,
    GameStateView,
    discretize_legal_actions,
    policy_to_game_action,
)
from src.nlhe.infoset import InfosetEncoder, _parse_universal_poker_state


N_DISCRETE_ACTIONS = len(DiscreteAction)  # 7


# ----- Networks -----

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int):
        super().__init__()
        dims = [in_dim] + hidden + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ----- Reservoir buffer -----

@dataclass
class ReservoirBuffer:
    capacity: int
    rng: random.Random
    features: list[np.ndarray] = field(default_factory=list)
    targets: list[np.ndarray] = field(default_factory=list)
    legal_masks: list[np.ndarray] = field(default_factory=list)
    iters: list[int] = field(default_factory=list)
    n_seen: int = 0

    def add(self, feature: np.ndarray, target: np.ndarray, legal_mask: np.ndarray, iteration: int) -> None:
        self.n_seen += 1
        if len(self.features) < self.capacity:
            self.features.append(feature)
            self.targets.append(target)
            self.legal_masks.append(legal_mask)
            self.iters.append(iteration)
        else:
            j = self.rng.randrange(self.n_seen)
            if j < self.capacity:
                self.features[j] = feature
                self.targets[j] = target
                self.legal_masks[j] = legal_mask
                self.iters[j] = iteration

    def __len__(self) -> int:
        return len(self.features)

    def sample_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self.features)
        idxs = [self.rng.randrange(n) for _ in range(min(batch_size, n))]
        feats = np.stack([self.features[i] for i in idxs])
        targs = np.stack([self.targets[i] for i in idxs])
        masks = np.stack([self.legal_masks[i] for i in idxs])
        its = np.array([self.iters[i] for i in idxs], dtype=np.int64)
        return (
            torch.from_numpy(feats).float(),
            torch.from_numpy(targs).float(),
            torch.from_numpy(masks).float(),
            torch.from_numpy(its),
        )


# ----- Helpers -----

def _build_game_state_view(state: Any, starting_stack: int) -> GameStateView:
    """Translate an OpenSpiel state into our GameStateView."""
    parsed = _parse_universal_poker_state(state)
    money = parsed["money"]
    player = parsed["player"]
    pot = parsed["pot"]
    my_money = money[player] if player < len(money) else 0
    opp_money = money[1 - player] if (1 - player) < len(money) else 0
    my_committed = starting_stack - my_money
    opp_committed = starting_stack - opp_money
    to_call = max(0, opp_committed - my_committed)
    effective_stack = min(my_money, opp_money)

    legal = state.legal_actions()
    legal_set = set(legal)
    # Min/max bet from the int legal actions (excluding fold=0, call=1).
    bet_actions = [a for a in legal if a >= 2]
    min_bet = min(bet_actions) if bet_actions else 0
    max_bet = max(bet_actions) if bet_actions else 0

    return GameStateView(
        pot=pot,
        to_call=to_call,
        effective_stack=effective_stack,
        min_bet=min_bet,
        max_bet=max_bet,
        legal_fold=(0 in legal_set),
        legal_call=(1 in legal_set),
    )


def _strategy_from_advantages(adv: np.ndarray, legal_mask: np.ndarray) -> np.ndarray:
    """Regret-matching+: positive advantages normalized; uniform on legal if all <= 0."""
    pos = np.maximum(adv, 0.0) * legal_mask
    s = pos.sum()
    if s > 0:
        return pos / s
    # Fall back to uniform over legal actions.
    legal_count = legal_mask.sum()
    if legal_count > 0:
        return legal_mask / legal_count
    return np.zeros_like(adv)


# ----- Solver -----

@dataclass
class TrainConfig:
    hidden_dim: list[int] = field(default_factory=lambda: [64, 64])
    n_iterations: int = 20
    traversals_per_iter: int = 100
    train_steps_per_iter: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    buffer_capacity: int = 50000
    starting_stack: int = 20000
    bucket_runouts: int = 50
    max_traversal_depth: int = 200  # safety cap
    seed: int = 2026
    # DCFR variants - see Brown & Sandholm 2019, simplified single-exponent form.
    # "vanilla":    all training samples weighted equally (current behavior)
    # "linear":     sample weight = t_i / T  (equivalent to dcfr_exponent=1.0)
    # "discounted": sample weight = (t_i / T) ** dcfr_exponent
    # t_i = iteration at which the sample was added; T = current iter.
    # Weights are normalized per-batch to sum to batch_size, preserving gradient scale.
    cfr_variant: str = "vanilla"
    dcfr_exponent: float = 1.0


class DeepCFRSolver:
    def __init__(
        self,
        game: Any,
        abstraction: Abstraction,
        config: TrainConfig,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.game = game
        self.abstraction = abstraction
        self.cfg = config
        self.log = logger or print

        self.rng = random.Random(config.seed)
        torch.manual_seed(config.seed)

        self.encoder = InfosetEncoder(
            abstraction=abstraction,
            max_buckets=200,
            starting_stack=config.starting_stack,
            bucket_runouts=config.bucket_runouts,
        )
        in_dim = self.encoder.feature_dim

        # Networks: one advantage and one strategy net per player.
        self.adv_nets = [
            MLP(in_dim, config.hidden_dim, N_DISCRETE_ACTIONS) for _ in range(2)
        ]
        self.strat_nets = [
            MLP(in_dim, config.hidden_dim, N_DISCRETE_ACTIONS) for _ in range(2)
        ]
        self.adv_opts = [
            optim.Adam(net.parameters(), lr=config.learning_rate) for net in self.adv_nets
        ]
        self.strat_opts = [
            optim.Adam(net.parameters(), lr=config.learning_rate) for net in self.strat_nets
        ]

        # Reservoir buffers: one advantage buffer per player, one shared strategy buffer.
        self.adv_buffers = [
            ReservoirBuffer(capacity=config.buffer_capacity, rng=random.Random(config.seed + 1 + p))
            for p in range(2)
        ]
        self.strat_buffer = ReservoirBuffer(
            capacity=config.buffer_capacity, rng=random.Random(config.seed + 100)
        )

        # GPU support: move networks to CUDA if available, else CPU.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for net in self.adv_nets + self.strat_nets:
            net.to(self.device)
        self.log(f"solver device: {self.device}")
        self.iteration = 0

    def _current_strategy(self, state: Any, traverser: int, t: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """Compute the regret-matching strategy at the current state via the advantage net.

        Returns:
            (strategy, legal_mask, discrete_to_chip_map)
        """
        view = _build_game_state_view(state, self.cfg.starting_stack)
        legal_chip = state.legal_actions()
        discrete_to_chip = discretize_legal_actions(legal_chip, view)
        legal_mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
        for da in discrete_to_chip:
            legal_mask[int(da)] = 1.0

        feat = self.encoder.encode(state, rng=self.rng)
        player = state.current_player()

        # Advantage net prediction at this infoset.
        self.adv_nets[player].eval()
        with torch.no_grad():
            adv = self.adv_nets[player](torch.from_numpy(feat).float().unsqueeze(0).to(self.device)).cpu().numpy()[0]
        # Apply regret matching+.
        strat = _strategy_from_advantages(adv, legal_mask)
        return strat, legal_mask, discrete_to_chip

    def _traverse(self, state: Any, traverser: int, depth: int = 0) -> float:
        """External-sampling CFR traversal. Returns the value to the traverser at this state."""
        if state.is_terminal():
            return float(state.returns()[traverser])
        if depth > self.cfg.max_traversal_depth:
            return 0.0
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            a = self.rng.choices(actions, weights=probs)[0]
            new_state = state.child(int(a))
            return self._traverse(new_state, traverser, depth + 1)

        current_player = state.current_player()
        strat, legal_mask, discrete_to_chip = self._current_strategy(state, traverser, self.iteration)

        if current_player == traverser:
            # Traverser node: evaluate ALL legal actions to compute counterfactual values.
            values_per_action = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
            feat = self.encoder.encode(state, rng=self.rng)
            for da, chip_action in discrete_to_chip.items():
                if chip_action is None:
                    continue
                # Recurse on this action.
                child = state.child(int(chip_action))
                values_per_action[int(da)] = self._traverse(child, traverser, depth + 1)
            ev = float((strat * values_per_action).sum())
            # Counterfactual regrets: r(a) = v(a) - v(strategy).
            # Normalize by starting_stack to keep MSE loss on O(1) scale instead
            # of O(stack^2) scale; this improves gradient conditioning.
            regrets = (values_per_action - ev) * legal_mask / max(self.cfg.starting_stack, 1)
            self.adv_buffers[traverser].add(feat, regrets.copy(), legal_mask.copy(), self.iteration)
            return ev
        else:
            # Opponent node: sample one action and recurse.
            feat = self.encoder.encode(state, rng=self.rng)
            # Add the current strategy to the strategy buffer (it's what we want to train toward).
            self.strat_buffer.add(feat, strat.copy(), legal_mask.copy(), self.iteration)
            # Sample an action proportional to current strategy, restricted to legal.
            probs = strat / max(strat.sum(), 1e-8)
            chosen = self.rng.choices(range(N_DISCRETE_ACTIONS), weights=probs.tolist(), k=1)[0]
            # Map discrete action to chip action.
            da = DiscreteAction(chosen)
            chip_action = discrete_to_chip.get(da)
            if chip_action is None:
                # Sampled an illegal action; pick uniform from legal as fallback.
                legal_items = list(discrete_to_chip.items())
                da, chip_action = self.rng.choice(legal_items)
            child = state.child(int(chip_action))
            return self._traverse(child, traverser, depth + 1)

    def _dcfr_weights(self, iters: torch.Tensor) -> torch.Tensor | None:
        """Per-sample DCFR weights for a batch.

        Vanilla -> None (caller falls back to unweighted mean).
        Linear  -> w_i = t_i / T
        Discounted -> w_i = (t_i / T) ** dcfr_exponent

        Weights normalized so they sum to batch_size, preserving gradient scale
        relative to the unweighted case. T defaults to max(1, current iter) to
        keep weights bounded in [0, 1] during the very first iteration.
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
        # Normalize to sum to batch size, preserving gradient scale.
        s = w.sum().clamp(min=1e-8)
        return w * (w.shape[0] / s)

    def _train_advantage_net(self, player: int) -> float:
        """One training pass on the advantage network. Returns mean loss or NaN if buffer too small."""
        buf = self.adv_buffers[player]
        if len(buf) < self.cfg.batch_size:
            return float("nan")
        net = self.adv_nets[player]
        opt = self.adv_opts[player]
        net.train()
        total_loss = 0.0
        for _ in range(self.cfg.train_steps_per_iter):
            feats, targets, masks, iters = buf.sample_batch(self.cfg.batch_size)
            feats = feats.to(self.device); targets = targets.to(self.device); masks = masks.to(self.device)
            iters = iters.to(self.device)
            preds = net(feats)
            # MSE on the legal subset only, per-sample then weighted.
            per_sample = ((preds - targets) ** 2 * masks).sum(dim=1)
            weights = self._dcfr_weights(iters)
            loss = (weights * per_sample).mean() if weights is not None else per_sample.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        return total_loss / self.cfg.train_steps_per_iter

    def _train_strategy_net(self, player: int) -> float:
        """Train the player's strategy net on the shared strategy buffer."""
        buf = self.strat_buffer
        if len(buf) < self.cfg.batch_size:
            return float("nan")
        net = self.strat_nets[player]
        opt = self.strat_opts[player]
        net.train()
        total_loss = 0.0
        for _ in range(self.cfg.train_steps_per_iter):
            feats, targets, masks, iters = buf.sample_batch(self.cfg.batch_size)
            feats = feats.to(self.device); targets = targets.to(self.device); masks = masks.to(self.device)
            iters = iters.to(self.device)
            # Predict softmax over actions; train via KL between target and softmax(pred).
            logits = net(feats)
            logits = logits - logits.max(dim=1, keepdim=True).values  # numerical stability
            exp_l = torch.exp(logits) * masks
            denom = exp_l.sum(dim=1, keepdim=True).clamp(min=1e-8)
            probs = exp_l / denom
            # KL(target || probs) approximated as -sum(target * log(probs+eps)), per-sample then weighted.
            per_sample = -(targets * torch.log(probs + 1e-8) * masks).sum(dim=1)
            weights = self._dcfr_weights(iters)
            loss = (weights * per_sample).mean() if weights is not None else per_sample.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        return total_loss / self.cfg.train_steps_per_iter

    def save_checkpoint(self, path: str | Path) -> None:
        """Persist solver state to disk for resumable training.

        Saves: networks (4), optimizers (4), reservoir buffers (3), RNG state,
        torch RNG state, current iteration. Single file, torch.save format.
        """
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "iteration": self.iteration,
            "adv_nets": [n.state_dict() for n in self.adv_nets],
            "strat_nets": [n.state_dict() for n in self.strat_nets],
            "adv_opts": [o.state_dict() for o in self.adv_opts],
            "strat_opts": [o.state_dict() for o in self.strat_opts],
            "adv_buffers": [
                {"features": b.features, "targets": b.targets,
                 "legal_masks": b.legal_masks, "iters": b.iters,
                 "n_seen": b.n_seen, "rng_state": b.rng.getstate()}
                for b in self.adv_buffers
            ],
            "strat_buffer": {
                "features": self.strat_buffer.features,
                "targets": self.strat_buffer.targets,
                "legal_masks": self.strat_buffer.legal_masks,
                "iters": self.strat_buffer.iters,
                "n_seen": self.strat_buffer.n_seen,
                "rng_state": self.strat_buffer.rng.getstate(),
            },
            "rng_state": self.rng.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "config_dict": self.cfg.__dict__,
        }, str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore solver state from a checkpoint file."""
        ckpt = torch.load(str(path), weights_only=False, map_location=self.device)
        self.iteration = ckpt["iteration"]
        for i, sd in enumerate(ckpt["adv_nets"]):
            self.adv_nets[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["strat_nets"]):
            self.strat_nets[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["adv_opts"]):
            self.adv_opts[i].load_state_dict(sd)
        for i, sd in enumerate(ckpt["strat_opts"]):
            self.strat_opts[i].load_state_dict(sd)
        for i, b_data in enumerate(ckpt["adv_buffers"]):
            self.adv_buffers[i].features = b_data["features"]
            self.adv_buffers[i].targets = b_data["targets"]
            self.adv_buffers[i].legal_masks = b_data["legal_masks"]
            self.adv_buffers[i].n_seen = b_data["n_seen"]
            self.adv_buffers[i].rng.setstate(b_data["rng_state"])
            if "iters" in b_data:
                self.adv_buffers[i].iters = b_data["iters"]
            else:
                # Pre-DCFR checkpoint. Refuse to resume non-vanilla; permit vanilla.
                if self.cfg.cfr_variant != "vanilla":
                    raise RuntimeError(
                        f"Checkpoint predates DCFR support (no per-sample iter tracking) "
                        f"but cfr_variant={self.cfg.cfr_variant!r}. Either resume with "
                        f"cfr_variant='vanilla' or start fresh."
                    )
                self.adv_buffers[i].iters = [1] * len(b_data["features"])
        sb = ckpt["strat_buffer"]
        self.strat_buffer.features = sb["features"]
        self.strat_buffer.targets = sb["targets"]
        self.strat_buffer.legal_masks = sb["legal_masks"]
        self.strat_buffer.n_seen = sb["n_seen"]
        self.strat_buffer.rng.setstate(sb["rng_state"])
        if "iters" in sb:
            self.strat_buffer.iters = sb["iters"]
        else:
            if self.cfg.cfr_variant != "vanilla":
                raise RuntimeError(
                    f"Checkpoint predates DCFR support (no per-sample iter tracking) "
                    f"but cfr_variant={self.cfg.cfr_variant!r}. Either resume with "
                    f"cfr_variant='vanilla' or start fresh."
                )
            self.strat_buffer.iters = [1] * len(sb["features"])
        self.rng.setstate(ckpt["rng_state"])
        try:
            torch.set_rng_state(ckpt["torch_rng_state"])
        except (TypeError, RuntimeError) as e:
            # PyTorch RNG state format may differ across versions / map_location.
            # Inference doesn't need RNG continuity, so skip restoration.
            print(f"Note: skipping torch RNG state restore ({type(e).__name__}: {e})")

    def train(self, checkpoint_dir: str | Path | None = None, checkpoint_every: int = 5) -> dict:
        """Run the full CFR training loop. Returns a dict of per-iteration metrics.

        If checkpoint_dir is provided, save state every `checkpoint_every` iterations
        and on completion. To resume, instantiate the solver and call load_checkpoint
        before train().
        """
        from pathlib import Path
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        start_iter = self.iteration + 1
        metrics = {
            "iter": [], "time": [], "adv_loss_0": [], "adv_loss_1": [],
            "strat_loss_0": [], "strat_loss_1": [],
            "adv_buf_0": [], "adv_buf_1": [], "strat_buf": [],
        }
        t_start = time.time()
        for it in range(start_iter, self.cfg.n_iterations + 1):
            self.iteration = it
            t_it = time.time()
            traverser = (it - 1) % 2

            # Traversals.
            self.encoder.reset_cache()
            for _ in range(self.cfg.traversals_per_iter):
                state = self.game.new_initial_state()
                self._traverse(state, traverser=traverser, depth=0)

            # Train both nets after each traversal batch.
            adv_loss = [float("nan"), float("nan")]
            adv_loss[traverser] = self._train_advantage_net(traverser)
            strat_loss = [self._train_strategy_net(0), self._train_strategy_net(1)]

            elapsed = time.time() - t_it
            metrics["iter"].append(it)
            metrics["time"].append(elapsed)
            metrics["adv_loss_0"].append(adv_loss[0])
            metrics["adv_loss_1"].append(adv_loss[1])
            metrics["strat_loss_0"].append(strat_loss[0])
            metrics["strat_loss_1"].append(strat_loss[1])
            metrics["adv_buf_0"].append(len(self.adv_buffers[0]))
            metrics["adv_buf_1"].append(len(self.adv_buffers[1]))
            metrics["strat_buf"].append(len(self.strat_buffer))

            if checkpoint_dir is not None and (it % checkpoint_every == 0 or it == self.cfg.n_iterations):
                ckpt_path = checkpoint_dir / f"ckpt_iter_{it:04d}.pt"
                self.save_checkpoint(ckpt_path)
                self.log(f"  saved checkpoint: {ckpt_path}")
            self.log(
                f"iter {it:>3}/{self.cfg.n_iterations}  "
                f"trav={traverser}  "
                f"adv={adv_loss[traverser] if not math.isnan(adv_loss[traverser]) else 'nan':>8}  "
                f"strat0={strat_loss[0] if not math.isnan(strat_loss[0]) else 'nan':>8}  "
                f"strat1={strat_loss[1] if not math.isnan(strat_loss[1]) else 'nan':>8}  "
                f"buf=(adv0={len(self.adv_buffers[0])}, adv1={len(self.adv_buffers[1])}, "
                f"strat={len(self.strat_buffer)})  "
                f"{elapsed:.1f}s"
            )

        total = time.time() - t_start
        self.log(f"=== total: {total/60:.1f} min ===")
        return metrics
