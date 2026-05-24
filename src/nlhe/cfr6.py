"""External-sampling CFR traversal for 6-max NLHE (Phase 4e.3b).

Builds on:
  - src/nlhe/networks6.py PlayerNetworks6Max (4e.3a)  — buffers to write regret samples
  - src/nlhe/infoset6.py InfosetEncoder6Max (4d)      — state -> 236-dim features
  - src/nlhe/icm_returns.py icm_adjust_returns (4e.2) — chip-EV -> ICM-EV at terminals
  - src/nlhe/actions.py DiscreteAction, ...           — 7-action discrete abstraction
  - src/nlhe/solver.py _strategy_from_advantages      — regret-matching+ helper

Pattern mirrors src/nlhe/solver.py's _traverse (HUNL reference), generalized
from 2-player chip-EV to 6-player ICM-EV.

What this module IS:
  - One recursive function `traverse_6max` that walks the game tree from a
    given state, samples opponent actions, enumerates traverser actions,
    accumulates counterfactual regret samples into the traverser's buffer.
  - A `CFR6MaxContext` dataclass bundling the per-iteration deps (encoder,
    networks, ICM parameters, iteration counter) so the recursive signature
    stays short.

What this module IS NOT:
  - A training loop. That's Phase 4e.3c.
  - A strategy / average-policy net. PlayerNetworks6Max only carries
    advantage nets right now; average-strategy approximation comes in
    a later 4e.3 sub-phase.
  - A subgame solver (B1) or within-match adaptation (C1).
  - HUNL-compatible. HUNL has its own solver.py and infoset.py.

Important correctness notes (re-read before modifying):

  1. Sample from CURRENT strategy (regret-matched from the seat's advantage
     net), NOT from an average strategy. The average strategy is what gets
     deployed, but external-sampling CFR's sampling mechanism uses the
     current iterate's regret-matched policy.

  2. At a traverser node, enumerate every legal discrete action and recurse
     on each. The counterfactual value v(a) at action a is the expected
     utility for the traverser GIVEN they take action a deterministically —
     i.e., the traverser's own reach contribution at this node is treated as
     1, not multiplied by sigma(a). The traverser's strategy ONLY enters
     when computing ev = sum_a sigma(a) * v(a) at the SAME node.

  3. External sampling: enumerate at traverser, sample at opponents. At an
     opponent node, draw one action from their current strategy and recurse
     on that single action only.

  4. ICM transformation is applied EXCLUSIVELY at terminal states. Internal
     nodes already deal in equity units once the terminal transform happens
     — standard CFR backups (weighted sums and differences) preserve that
     equity-space interpretation.

  5. Regret samples are NOT divided by starting_stack here (the HUNL solver
     does that division). ICM-adjusted utilities are bounded by the payout
     structure (e.g. max equity ~2.0 buy-ins for Double Up), so regrets are
     already on O(1) scale and MSE gradients are well-conditioned without
     normalization. HUNL needed the divide because chip-EV regrets ran into
     the thousands.

  6. Sample addition is restricted to the traverser's buffer at the
     traverser's own decision nodes. Opponent nodes write nothing to any
     buffer in this module — that's the future strategy-net's job, not 4e.3b's.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from src.nlhe.actions import (
    DiscreteAction,
    GameStateView,
    discretize_legal_actions,
)
from src.nlhe.icm_returns import icm_adjust_returns
from src.nlhe.infoset6 import InfosetEncoder6Max, parse_state_6max, parse_state_repeated_6max
from src.nlhe.networks6 import PlayerNetworks6Max, N_DISCRETE_ACTIONS, NUM_SEATS_6MAX
from src.nlhe.solver import _strategy_from_advantages


@dataclass
class CFR6MaxContext:
    """Per-iteration context bundling deps that don't change across a single traversal.

    The training loop in Phase 4e.3c constructs this once per iteration and
    passes it through the recursion. Encoder cache and buffer state mutate
    during the traversal; numeric fields are read-only within a traversal.

    Args:
        policy_nets: 6-seat advantage networks + reservoir buffers (4e.3a).
        encoder: 6-max infoset feature encoder (4d).
        starting_stacks: per-seat chip count at the START of this hand
            (before blinds posted). Length 6. Used by icm_adjust_returns
            at terminals.
        payouts: tournament prize structure by finish position. Length K <= 6.
            Use sng_payouts_6max_double_up() or sng_payouts_6max_standard()
            from icm.py.
        iteration: current training iteration index. Stored on each buffer
            sample (so DCFR weighting in 4e.3c training can read it).
        max_depth: safety cap on recursion depth. Healthy 6-max games rarely
            exceed depth 100; the cap is here to detect infinite loops, not
            to truncate real play.
    """
    policy_nets: PlayerNetworks6Max
    encoder: InfosetEncoder6Max
    starting_stacks: Sequence[int]
    payouts: Sequence[float]
    iteration: int = 0
    max_depth: int = 500
    num_paid: int = 3

    def __post_init__(self) -> None:
        if len(self.starting_stacks) != NUM_SEATS_6MAX:
            raise ValueError(
                f"starting_stacks must have {NUM_SEATS_6MAX} entries, "
                f"got {len(self.starting_stacks)}"
            )
        if not self.payouts:
            raise ValueError("payouts must be non-empty")
        if self.num_paid < 1 or self.num_paid > NUM_SEATS_6MAX:
            raise ValueError(
                f"num_paid must be in [1, {NUM_SEATS_6MAX}], got {self.num_paid}"
            )


def _build_view_6max(state: Any, parsed: dict) -> GameStateView:
    """Build a GameStateView from a parsed 6-max universal_poker state.

    Takes the already-parsed dict to avoid re-running observation_string()
    twice (the caller needs `parsed` for other reasons in the same recursive
    frame).

    The 6-max view differs from HUNL's _build_game_state_view in that
    `to_call` and `effective_stack` are derived against the largest active
    opponent, not "the one other player". For action discretization
    (the actual consumer of this view), only pot/min_bet/max_bet/legal_*
    are read; to_call and effective_stack are informational.
    """
    cp = parsed["current_player"]
    money = parsed["money"]
    contribution = parsed["contribution"]
    pot = parsed["pot"]

    my_money = money[cp] if cp < len(money) else 0
    max_contrib = max(contribution) if contribution else 0
    my_contrib = contribution[cp] if cp < len(contribution) else 0
    to_call = max(0, max_contrib - my_contrib)

    # Effective stack: hero vs largest active opponent. "Active" here is
    # approximated by stack > 0; folded-but-not-busted players still appear
    # active by this test, but their committed chips are still in the pot
    # so the bet sizing math is unaffected.
    opp_stacks = [money[i] for i in range(len(money))
                  if i != cp and money[i] > 0]
    if opp_stacks:
        effective_stack = min(my_money, max(opp_stacks))
    else:
        effective_stack = my_money

    legal = state.legal_actions()
    legal_set = set(legal)
    # OpenSpiel universal_poker: 0=fold, 1=call/check, N>=2=bet-to-N-chips.
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



def is_tournament_terminal(state, num_paid=3):
    """Check whether the tournament is terminal for top-N-equal-pay format.

    For Double Up 6-max (top-3-equal-pay), the tournament effectively ends
    when player #4 busts. OpenSpiel's repeated_poker only marks the state
    terminal when 1 player has all chips. This helper checks whether the
    number of players with nonzero stacks is at-or-below num_paid.
    """
    if not hasattr(state, "stacks"):
        return False
    stacks = state.stacks()
    alive = sum(1 for s in stacks if s > 0)
    return alive <= num_paid


def compute_icm_payouts(stacks, num_paid=3, payout_per_seat=2.0, buy_in_units=1.0):
    """Compute per-seat payouts in normalized units at tournament terminal.

    For top-N-equal-pay (Double Up), each cashing player receives an equal
    share. Payouts are normalized to buy-in units: a cashing player
    receives +1.0, a busted player receives -1.0.
    """
    profit_per_cash = payout_per_seat - buy_in_units
    loss_per_bust = -buy_in_units
    return [profit_per_cash if s > 0 else loss_per_bust for s in stacks]


def traverse_6max(
    state: Any,
    traversing_player: int,
    ctx: CFR6MaxContext,
    rng: random.Random,
    depth: int = 0,
    opponent_policy_override: Optional[Any] = None,
) -> float:
    """One external-sampling CFR traversal for 6-max NLHE.

    Recursively walks the game tree from `state`. At terminals returns the
    ICM-adjusted utility for `traversing_player`. At chance nodes samples
    one outcome and recurses. At decision nodes:
      - If current player == traversing_player: enumerate all legal discrete
        actions, recurse on each, compute counterfactual value and regrets,
        write a regret sample to the traverser's buffer, return the
        strategy-weighted expected value.
      - Otherwise: sample one action from the current player's current
        (regret-matched) strategy and recurse.

    Args:
        state: an OpenSpiel universal_poker state (typically obtained from
            game.new_initial_state(); chance nodes will be handled by the
            recursion).
        traversing_player: seat index in [0, 6) that this traversal is for.
            Regret samples are added to ctx.policy_nets.buffer_for(this
            player) at decision nodes where the current player matches.
        ctx: CFR6MaxContext bundle (encoder, networks, ICM params, iter).
        rng: random source used to sample chance outcomes and opponent
            actions. The caller controls reproducibility via this rng's seed.
        depth: current recursion depth (internal; callers should not pass).
        opponent_policy_override: optional Policy (per the protocol in
            scripts/eval_pool.py — exposes select_action(parsed, state, rng,
            mode) -> int chip-action). When set, all NON-traverser decision
            nodes encountered in this traversal route through the override
            instead of the self-play advantage net. Used by league play:
            the solver pre-samples one league opponent per traversal and
            threads it down. Default None preserves pre-league behavior
            bit-for-bit; per-traversal granularity means all 5 opponent
            seats share the same override within one traversal.

    Returns:
        The counterfactual value to `traversing_player` at this node, in
        ICM-equity units.

    Raises:
        IndexError: if traversing_player is out of [0, 6).
    """
    if not (0 <= traversing_player < NUM_SEATS_6MAX):
        raise IndexError(
            f"traversing_player {traversing_player} out of range "
            f"[0, {NUM_SEATS_6MAX})"
        )

    # ---- Tournament terminal: bubble has burst (alive players <= num_paid).
    # For multi-hand tournaments, the game continues past the bubble; for
    # top-N-equal-pay formats we stop earlier and compute ICM payouts on
    # current stacks. Single-hand games do not have stacks(); the helper
    # returns False in that case and we fall through to state.is_terminal().
    if is_tournament_terminal(state, num_paid=ctx.num_paid):
        stacks = state.stacks()
        payout_per_seat = float(ctx.payouts[0]) if ctx.payouts else 2.0
        icm_payouts = compute_icm_payouts(
            stacks=stacks,
            num_paid=ctx.num_paid,
            payout_per_seat=payout_per_seat,
            buy_in_units=1.0,
        )
        return float(icm_payouts[traversing_player])

    # ---- Terminal: ICM-adjust the chip-EV returns and pick the traverser's slice.
    if state.is_terminal():
        chip_returns = list(state.returns())
        icm_returns = icm_adjust_returns(
            chip_returns=chip_returns,
            starting_stacks=ctx.starting_stacks,
            payouts=ctx.payouts,
        )
        return float(icm_returns[traversing_player])

    # ---- Depth cap (safety; healthy 6-max games don't get near this).
    if depth > ctx.max_depth:
        return 0.0

    # ---- Chance node: sample one outcome, recurse.
    if state.is_chance_node():
        outcomes = state.chance_outcomes()
        actions, probs = zip(*outcomes)
        sampled = rng.choices(actions, weights=probs, k=1)[0]
        return traverse_6max(
            state.child(int(sampled)),
            traversing_player,
            ctx,
            rng,
            depth + 1,
            opponent_policy_override=opponent_policy_override,
        )

    # ---- Decision node.
    cp = state.current_player()
    # Repeated_poker states expose dealer_seat(); single-hand states don't.
    if hasattr(state, "dealer_seat"):
        parsed = parse_state_repeated_6max(state)
    else:
        parsed = parse_state_6max(state)

    # ---- League-play short-circuit: at NON-traverser nodes with an override,
    # the opponent's action comes from the override policy. We skip the
    # self-play machinery (encoder.encode_from_parsed + advantage-net forward
    # pass + regret-matching) entirely — the override is responsible for its
    # own feature encoding and action selection. Per the Policy protocol in
    # scripts/eval_pool.py, select_action returns an OpenSpiel chip action.
    if opponent_policy_override is not None and cp != traversing_player:
        chip_action = opponent_policy_override.select_action(
            parsed, state, rng, mode="sample"
        )
        child = state.child(int(chip_action))
        return traverse_6max(
            child,
            traversing_player,
            ctx,
            rng,
            depth + 1,
            opponent_policy_override=opponent_policy_override,
        )

    view = _build_view_6max(state, parsed)

    legal_chip = list(state.legal_actions())
    discrete_to_chip = discretize_legal_actions(legal_chip, view)

    legal_mask = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
    for da in discrete_to_chip:
        legal_mask[int(da)] = 1.0

    if legal_mask.sum() == 0:
        # Shouldn't happen — FOLD and CALL always map to legal chip actions
        # when present in state.legal_actions(). Treat as terminal-of-zero
        # if it ever does (e.g. an unexpected universal_poker state shape).
        return 0.0

    feat = ctx.encoder.encode_from_parsed(parsed, rng=rng)

    # Current strategy at the acting player: regret-matched (RM+) from
    # their advantage net's output, masked to legal actions.
    adv = ctx.policy_nets.predict_advantages(seat=cp, features=feat)
    strat = _strategy_from_advantages(adv, legal_mask)

    if cp == traversing_player:
        # Traverser node: enumerate ALL legal discrete actions, recurse on each.
        values_per_action = np.zeros(N_DISCRETE_ACTIONS, dtype=np.float32)
        for da, chip_action in discrete_to_chip.items():
            if chip_action is None:
                continue  # discrete action not mappable to any legal chip action
            child = state.child(int(chip_action))
            values_per_action[int(da)] = traverse_6max(
                child,
                traversing_player,
                ctx,
                rng,
                depth + 1,
                opponent_policy_override=opponent_policy_override,
            )

        # Expected value at this infoset under the CURRENT strategy.
        ev = float((strat * values_per_action).sum())

        # Counterfactual regrets r(a) = v(a) - v(strategy), zeroed on
        # illegal actions. NOT normalized by starting_stack — ICM-adjusted
        # utilities are bounded by payouts (typically <= 3.9 buy-ins), so
        # regrets are already O(1).
        regrets = (values_per_action - ev) * legal_mask

        ctx.policy_nets.buffer_for(traversing_player).add(
            feat.copy(),
            regrets.copy(),
            legal_mask.copy(),
            ctx.iteration,
        )
        return ev
    else:
        # Opponent node: sample ONE action from their current strategy, recurse.
        # _strategy_from_advantages guarantees strat is zero on illegal
        # actions when legal_count > 0, so the sample is always legal.
        s = float(strat.sum())
        if s <= 0:
            # Defensive: this can only happen if legal_mask.sum() == 0,
            # which we already handled above. Treat as no-op.
            return 0.0
        probs = (strat / s).astype(np.float64)
        chosen_idx = rng.choices(
            range(N_DISCRETE_ACTIONS),
            weights=probs.tolist(),
            k=1,
        )[0]
        da = DiscreteAction(chosen_idx)
        chip_action = discrete_to_chip.get(da)
        if chip_action is None:
            # Belt-and-suspenders: shouldn't trigger (strat is masked).
            legal_items = list(discrete_to_chip.items())
            da, chip_action = rng.choice(legal_items)
        child = state.child(int(chip_action))
        return traverse_6max(
            child,
            traversing_player,
            ctx,
            rng,
            depth + 1,
            opponent_policy_override=opponent_policy_override,
        )
