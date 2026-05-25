"""Depth-limited subgame tree builder for Track B1 (Pluribus-style real-time
solving).

This is the first concrete module of B1c. It constructs the game tree rooted
at a given decision point, traversing forward up to a configurable depth,
and produces a Node-tree data structure that:

  - Captures every reachable game state up to the depth limit
  - Aggregates states into information sets (for CFR over the subgame)
  - Subsamples chance nodes (board card deals) for tractability
  - Marks leaf nodes (depth-limited terminals) as needing leaf evaluation

The tree this module produces is the INPUT to:
  - The leaf evaluator (sub-step 2 of B1) — assigns expected payoffs to leaves
  - The subgame CFR loop (sub-step 3 of B1) — solves the depth-limited game
  - Policy extraction (sub-step 4 of B1) — extracts hero's refined strategy

NOTHING IN THIS MODULE RUNS CFR. It only builds the tree.

Status: B1c step 1 — tree construction. Solver loop not yet present.

Design choices (with rationale):
  - Recursive Python objects. Easier to debug than flat arrays; profile-and-
    convert if speed matters.
  - Chance nodes subsample uniformly. Card abstraction integration (EMD
    bucketing) is a v2 optimization once the basic tree+solver works.
  - Depth measured in ACTIONS, not streets. Simpler to reason about and
    matches Brown/Sandholm depth-limited papers.
  - Infoset keys derived from infoset6.py encoding (already validated in
    Phase 4d). Lets us query the existing blueprint at leaves.

References:
  - Brown, Sandholm, Amos (NeurIPS 2018) "Depth-Limited Solving for
    Imperfect-Information Games" — sets the depth-limited framework
  - Pluribus, Brown & Sandholm (Science 2019) — extends to multi-player
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

log = logging.getLogger("subgame")


# ============================================================
# Node types
# ============================================================

class NodeKind(Enum):
    """What kind of node this is in the subgame tree."""
    DECISION = "decision"    # A player must act
    CHANCE = "chance"        # Board cards being dealt
    TERMINAL = "terminal"    # Hand reached natural end (showdown/fold-all)
    LEAF = "leaf"            # Depth limit hit; needs leaf evaluation


# ============================================================
# Node
# ============================================================

@dataclass
class SubgameNode:
    """A single node in the subgame tree.

    Each node is exactly one of: decision, chance, terminal, or leaf.
    Decision nodes have a current_player and one child per legal discrete
    action. Chance nodes have one child per sampled chance outcome.
    Terminal nodes carry the natural-game terminal payoffs. Leaf nodes
    carry NO payoffs — those are assigned later by the leaf evaluator.

    Attributes:
        kind: which kind of node this is.
        state: the underlying OpenSpiel state. Kept so children/payoffs
            can be derived. Note: states are mutable so we never modify
            them — only call read-only methods and create children.
        depth: action depth from root (root = 0). Used for depth-limit
            check.
        action_from_parent: the action that led to this node from its parent.
            For a child of a DECISION node this is the DiscreteAction value
            (an IntEnum, 0..6 — the durable representation; the raw chip int
            is re-derivable via discretize_legal_actions at the parent state).
            For a child of a CHANCE node this is the raw chance-outcome int.
            None at root.
        chance_prob: probability of this branch given the parent (chance
            nodes only). For decisions, weight comes from the strategy.
        current_player: at decision nodes, who acts. None elsewhere.
        infoset_key: at decision nodes, a hashable key identifying the
            information set (multiple decision nodes may share an infoset).
            None for chance/terminal/leaf nodes.
        terminal_returns: at terminal nodes, the per-player payoffs from
            state.returns(). None elsewhere.
        children: list of child SubgameNode. Empty for terminal and leaf.
        action_at_child: parallel to `children` — the action that produces
            each child. For chance nodes these are chance outcomes (ints);
            for decision nodes these are DiscreteAction values (ints), NOT
            raw chip ints. The chip int needed to replay via state.child()
            is derived on demand from discretize_legal_actions at this state.
        n_descendants: number of descendant nodes (computed after build).
        leaf_value: at LEAF nodes, the per-seat ICM-equity-delta 6-vector the
            leaf evaluator assigns (sub-step 2, src/nlhe/subgame_leaf.py). None
            until evaluated. Same units/shape as terminal_returns so sub-step 3
            backs up LEAF and TERMINAL nodes uniformly:
                value_vec = node.terminal_returns if node.is_terminal
                            else node.leaf_value
            Populated by subgame_leaf.evaluate_leaves(tree, ctx) before the CFR
            loop; a data field only (no behavior here).
    """
    kind: NodeKind
    state: Any
    depth: int
    action_from_parent: Optional[int] = None
    # One-STEP conditional probability of this branch given the parent (chance
    # nodes only), NOT a cumulative reach probability. Sub-step 3 multiplies
    # these along the path to recover reach.
    chance_prob: float = 1.0
    current_player: Optional[int] = None
    infoset_key: Optional[tuple] = None
    terminal_returns: Optional[list[float]] = None
    children: list["SubgameNode"] = field(default_factory=list)
    action_at_child: list[int] = field(default_factory=list)
    n_descendants: int = 0
    leaf_value: Optional[list[float]] = None

    @property
    def is_decision(self) -> bool:
        return self.kind == NodeKind.DECISION

    @property
    def is_chance(self) -> bool:
        return self.kind == NodeKind.CHANCE

    @property
    def is_terminal(self) -> bool:
        return self.kind == NodeKind.TERMINAL

    @property
    def is_leaf(self) -> bool:
        return self.kind == NodeKind.LEAF


# ============================================================
# Tree
# ============================================================

@dataclass
class SubgameTree:
    """Container for the constructed subgame tree.

    Attributes:
        root: the root node (always a decision node — we don't build
            subgames from chance or terminal states).
        all_nodes: every node in DFS-preorder. Useful for CFR loops.
        infoset_groups: mapping from infoset_key to the list of decision
            nodes that share that infoset. CFR operates on these groups.
            NOTE: under chance subsampling these groups are over the SAMPLED
            subgame, not the underlying game — information_state_string
            includes the sampled board, so decision nodes under different
            chance samples never merge into one infoset. This is intentional
            (matches Brown/Sandholm depth-limited solving); sub-step 3 should
            NOT try to "fix" it.
        n_decision_nodes / n_chance_nodes / n_terminal_nodes / n_leaf_nodes:
            counts by kind.
    """
    root: SubgameNode
    all_nodes: list[SubgameNode] = field(default_factory=list)
    infoset_groups: dict[tuple, list[SubgameNode]] = field(default_factory=dict)
    n_decision_nodes: int = 0
    n_chance_nodes: int = 0
    n_terminal_nodes: int = 0
    n_leaf_nodes: int = 0

    def summary(self) -> str:
        return (
            f"SubgameTree: {self.n_decision_nodes} decisions across "
            f"{len(self.infoset_groups)} infosets, "
            f"{self.n_chance_nodes} chance, "
            f"{self.n_terminal_nodes} terminal, "
            f"{self.n_leaf_nodes} leaves "
            f"({len(self.all_nodes)} total)"
        )


# ============================================================
# Tree construction
# ============================================================

def build_subgame_tree(
    state: Any,
    max_action_depth: int = 4,
    chance_samples_per_node: int = 8,
    rng=None,
    skip_unsupported_states: bool = True,
) -> SubgameTree:
    """Build a depth-limited subgame tree rooted at `state`.

    Args:
        state: an OpenSpiel state at a decision point. Must NOT be a chance
            or terminal state at the root — we want hero to be the root
            actor.
        max_action_depth: how many actions deep to expand the tree before
            cutting off into leaf nodes. Chance outcomes do NOT count as
            actions for this depth count.
        chance_samples_per_node: at each chance node, how many outcomes
            to sample (uniformly). Higher = more accurate, more expensive.
            8 is a reasonable default for flop deals.
        rng: random source for chance sampling. None = deterministic
            (use Python's `random` module globals).
        skip_unsupported_states: if True, return a tree with just a leaf
            root when the state doesn't have the expected interface
            (e.g., raw single-hand state instead of repeated_poker). If
            False, raise.

    Returns:
        SubgameTree.

    Raises:
        ValueError: if the root is a chance or terminal state.
    """
    import random as _random
    rng = rng or _random.Random()

    # Root cannot be chance or terminal — we build subgames around
    # specific hero decision points.
    if state.is_chance_node():
        raise ValueError("Cannot build subgame from a chance node — call advance "
                         "to the next decision point first")
    if state.is_terminal():
        raise ValueError("Cannot build subgame from a terminal state — there are "
                         "no decisions left")

    tree = SubgameTree(root=None)   # filled in below

    root = _build_node(
        state=state,
        depth=0,
        max_action_depth=max_action_depth,
        chance_samples_per_node=chance_samples_per_node,
        rng=rng,
        action_from_parent=None,
        chance_prob=1.0,
        tree=tree,
        skip_unsupported_states=skip_unsupported_states,
    )
    tree.root = root

    # Compute n_descendants for each node (DFS, bottom-up)
    _compute_descendants(root)

    return tree


def _build_node(
    state: Any,
    depth: int,
    max_action_depth: int,
    chance_samples_per_node: int,
    rng,
    action_from_parent: Optional[int],
    chance_prob: float,
    tree: SubgameTree,
    skip_unsupported_states: bool,
) -> SubgameNode:
    """Recursively build the tree from `state`."""
    # === Terminal state? ===
    if state.is_terminal():
        node = SubgameNode(
            kind=NodeKind.TERMINAL,
            state=state,
            depth=depth,
            action_from_parent=action_from_parent,
            chance_prob=chance_prob,
            terminal_returns=list(state.returns()),
        )
        tree.all_nodes.append(node)
        tree.n_terminal_nodes += 1
        return node

    # === Depth limit hit (action-depth, not counting chance)? Mark as leaf ===
    if depth >= max_action_depth:
        node = SubgameNode(
            kind=NodeKind.LEAF,
            state=state,
            depth=depth,
            action_from_parent=action_from_parent,
            chance_prob=chance_prob,
            current_player=(state.current_player()
                            if not state.is_chance_node() else None),
        )
        tree.all_nodes.append(node)
        tree.n_leaf_nodes += 1
        return node

    # === Chance node: COLLAPSE INTO A LEAF (chance is transparent). ===
    # Stage-5-B cost-bounding fix. Expanding chance into the tree caused a leaf
    # explosion at round-closing / all-in spots: chance branches ×chance_samples
    # AND does NOT consume the action-depth budget, so board deals compound across
    # streets within one depth-K budget (a depth-3 round-closing tree blew up to
    # 2560 leaves, 2048 of them chance). Depth now measures hero/opponent ACTIONS
    # only; a chance node becomes a LEAF whose state carries the pending board deal,
    # and the leaf evaluator's rollout draws the board as it plays forward (the
    # chance-safe parse from 03576eb handles current_player() == -1). The
    # `chance_samples_per_node` parameter is retained for API compatibility but is no
    # longer used for in-tree expansion.
    if state.is_chance_node():
        node = SubgameNode(
            kind=NodeKind.LEAF,
            state=state,
            depth=depth,
            action_from_parent=action_from_parent,
            chance_prob=chance_prob,
            current_player=None,  # chance node: no acting player
        )
        tree.all_nodes.append(node)
        tree.n_leaf_nodes += 1
        return node

    # === Decision node: enumerate legal DISCRETE actions ===
    # CRITICAL: we must NOT iterate raw state.legal_actions() here. Under
    # bettingAbstraction=fullgame (what six_max_sng uses) that returns ~10k
    # chip-amount ints, which would explode the tree to hundreds of leaves at
    # depth 1. The CFR walker (cfr6.traverse_6max) collapses those to the
    # 7-action DiscreteAction abstraction via discretize_legal_actions; the
    # tree builder MUST enumerate that identical action set so the subgame the
    # solver sees matches the game the walker plays.
    cp = state.current_player()
    infoset_key = _infoset_key_from_state(state, cp, skip_unsupported_states)

    node = SubgameNode(
        kind=NodeKind.DECISION,
        state=state,
        depth=depth,
        action_from_parent=action_from_parent,
        chance_prob=chance_prob,
        current_player=cp,
        infoset_key=infoset_key,
    )
    tree.all_nodes.append(node)
    tree.n_decision_nodes += 1

    # Register in infoset groups
    if infoset_key is not None:
        tree.infoset_groups.setdefault(infoset_key, []).append(node)

    # Source of truth for which children exist: the discrete-action map,
    # built via the exact same path cfr6.traverse_6max uses (cfr6.py:333-336).
    discrete_to_chip = _discretize_at_decision(state)
    for da, chip_action in discrete_to_chip.items():
        # Mirror cfr6.py:359 — defensive skip. discretize_legal_actions never
        # actually returns None values, but keep the guard so the tree builder
        # and the walker stay byte-for-byte aligned if that ever changes.
        if chip_action is None:
            continue
        # The raw chip int is needed only to fetch the child state. We store
        # the DiscreteAction value (IntEnum -> int-compatible) as the durable
        # action label, not the chip int.
        child_state = state.child(int(chip_action))
        child = _build_node(
            state=child_state,
            depth=depth + 1,        # decisions DO count toward action depth
            max_action_depth=max_action_depth,
            chance_samples_per_node=chance_samples_per_node,
            rng=rng,
            action_from_parent=int(da),
            chance_prob=1.0,        # decisions are deterministic given strategy
            tree=tree,
            skip_unsupported_states=skip_unsupported_states,
        )
        node.children.append(child)
        node.action_at_child.append(int(da))

    return node


def _discretize_at_decision(state) -> dict:
    """Enumerate the discrete action set at a decision `state`.

    Mirrors cfr6.traverse_6max exactly (cfr6.py:307-336): dispatch on
    dealer_seat to pick the parser, build the GameStateView from the parsed
    dict via cfr6._build_view_6max, then discretize the raw legal chip
    actions against that view. Returns the {DiscreteAction: chip_int} mapping
    that is the single source of truth for a decision node's children.

    Importing cfr6 here (lazily) rather than at module top keeps the pure-data
    dataclass tests free of the torch dependency that cfr6 -> networks6/solver
    pulls in; tree construction needs open_spiel + torch on the pod anyway.
    """
    from src.nlhe.actions import discretize_legal_actions
    from src.nlhe.cfr6 import _build_view_6max
    from src.nlhe.infoset6 import parse_state_6max, parse_state_repeated_6max

    # repeated_poker states expose dealer_seat(); single-hand states don't.
    if hasattr(state, "dealer_seat"):
        parsed = parse_state_repeated_6max(state)
    else:
        parsed = parse_state_6max(state)
    view = _build_view_6max(state, parsed)
    legal_chip = list(state.legal_actions())
    return discretize_legal_actions(legal_chip, view)


def _infoset_key_from_state(state, current_player: int,
                             skip_unsupported_states: bool) -> Optional[tuple]:
    """Derive a hashable infoset key from a decision state.

    The key includes everything the current player knows: their private
    cards, the public board so far, the betting history, who's active,
    stack sizes, and their seat position.

    For v1 we use a simple string-based key derived from the OpenSpiel
    information_state_string. Future versions may use the infoset6.py
    encoded vector (bucketed and quantized) for finer aggregation.
    """
    try:
        info_string = state.information_state_string(current_player)
        return ("info", info_string)
    except Exception as e:
        if skip_unsupported_states:
            log.debug(f"Could not derive infoset key: {e}")
            return None
        raise


def _compute_descendants(root: SubgameNode) -> None:
    """Set n_descendants on every node, in-place, via iterative post-order DFS.

    Iterative (explicit stack) rather than recursive: sub-step 3 will build
    deeper trees and Python's default 1000-frame recursion limit would start
    tripping on a recursive walk. A node is finalized only after all its
    children are, so n_descendants reads are always already-computed:
        n_descendants(node) = sum over children c of (1 + n_descendants(c))
    """
    stack: list[tuple[SubgameNode, bool]] = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            node.n_descendants = sum(1 + c.n_descendants for c in node.children)
        else:
            # Re-push the node to finalize after its children, then push
            # children to be processed first.
            stack.append((node, True))
            for c in node.children:
                stack.append((c, False))


# ============================================================
# Tree utilities
# ============================================================

def iter_decision_nodes(tree: SubgameTree):
    """Yield every decision node in the tree."""
    for n in tree.all_nodes:
        if n.is_decision:
            yield n


def iter_leaf_nodes(tree: SubgameTree):
    """Yield every leaf (depth-limited) node in the tree."""
    for n in tree.all_nodes:
        if n.is_leaf:
            yield n


def iter_terminal_nodes(tree: SubgameTree):
    """Yield every terminal (natural-game-end) node in the tree."""
    for n in tree.all_nodes:
        if n.is_terminal:
            yield n


def tree_depth(tree: SubgameTree) -> int:
    """Return the maximum depth observed in the tree (action-count depth).

    Note: this counts only action-depth, not chance branches. A tree
    built with max_action_depth=4 may have node depths up to exactly 4.
    """
    return max((n.depth for n in tree.all_nodes), default=0)


def infoset_count(tree: SubgameTree) -> int:
    """Return the number of distinct information sets in the tree."""
    return len(tree.infoset_groups)
