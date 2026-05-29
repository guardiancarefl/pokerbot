"""Step 7 dashboard — Piece B: periodic in-training mini-eval against anchors.

Every `mini_eval_every` iters, snapshot the live solver's nets to a temp
checkpoint, load it via the existing CheckpointPolicy machinery, and run a
short head-to-head (ICM-equity-delta, the sub-step-6 metric) against a fixed
set of anchor opponents plus one rotating Shanky profile. Produces a strength
heartbeat during the long Step 7 run without waiting for the post-run eval.

Isolation: evaluate_matchup self-seeds its own random.Random(seed) from an int
(eval_pool.py:174). We pass seed = cfg.seed + 200 + iter_num, so the mini-eval
NEVER touches the solver's training rng (self.rng). Training-trajectory
bit-identity is preserved by construction (see test_mini_eval_uses_dedicated_rng).

Reuses, does not reimplement:
  - CheckpointPolicy / evaluate_matchup / play_one_hand_two_policies (eval_pool)
  - icm_adjust_returns (the equity-delta metric, via play_one_hand)
  - ShankyProfilePolicy (scripted_bots) for the rotating Shanky anchor

Anchor specs accept "name=path" (preferred — clean log names) or a bare path
(name derived from the path). _load_solver builds an inference-only solver from
the checkpoint's config_dict (tournament_structure_path=None, no league/
archetype/mini_eval pools), so loading the challenger and anchors is cheap.
"""
from __future__ import annotations

import os
from typing import Any, Optional

# Fixed path: assumes a single training run per host. Concurrent training
# runs on the same host would clobber each other's challenger snapshot.
# Add a PID/run-dir suffix if multi-run support becomes needed.
CHALLENGER_TMP = "/tmp/step7_mini_eval_challenger.pt"


# ANSI escape codes for the colored boxed format. Used only when the caller
# passes colored=True to format_eval_block; the clean ASCII path stays
# completely escape-free.
_ANSI_RESET = "\033[0m"
_ANSI_HDR = "\033[36m"        # cyan — header row
_ANSI_POS = "\033[32m"        # green — positive ICM lift
_ANSI_NEG = "\033[31m"        # red — negative ICM lift
_ANSI_DIM = "\033[2m"         # dim — placeholder line


def format_eval_block(
    iter_num: int,
    wall_s: float,
    n_hands: int,
    results: dict,
    prev_results: Optional[dict],
    structure: Any,
    colored: bool,
    placeholder_self_label: Optional[str] = None,
) -> str:
    """Format one mini-eval cycle's results as a multi-line string.

    Two output flavors:
      - colored=False: clean ASCII single-line records per anchor —
        backward-compatible with the format test_dashboard_self_anchor.py
        asserts. Format per line:
            [iter NNNN] lift_vs_<anchor>: +X.XXXX ICM (+Y.Y bb/100)  NNNh
            std=X.XXXX sigma=X.XX
        Placeholder line (when placeholder_self_label is set):
            [iter NNNN] lift_vs_<label>: (no prior checkpoint)
      - colored=True: boxed multi-line block with ANSI color highlighting
        for positive (green) / negative (red) ICM lift, cyan header, dim
        gray placeholder. Wide ═ box-drawing borders. Anchor names
        left-padded to a fixed width so values column-align.

    Args:
        iter_num, wall_s, n_hands: header metadata.
        results: {anchor_name: {"lift", "std", "sigma"}} as produced by
            run_mini_eval.
        prev_results: previous cycle's results dict (used for the rolling-
            delta column in the colored format; ignored when None or empty).
        structure: tournament structure for the bb/100 conversion (None
            disables the bb/100 column).
        colored: select ASCII (False) or ANSI/box (True) output.
        placeholder_self_label: when set, append a "(no prior checkpoint)"
            line for that label after the real records. Used at the first
            eval cycle when no prior checkpoint exists yet.

    Returns:
        Multi-line string. Each non-empty line is a complete log record
        (the caller writes the string verbatim — no extra newlines added).
    """
    lines: list = []

    if colored:
        # Header band: cyan ═ borders, "iter NNNN — eval (X.Xmin, NNNNh/opp)"
        header_inner = (
            f" iter {iter_num} — eval "
            f"({wall_s/60:.1f}min, {n_hands}h/opp) "
        )
        border = "═" * 3
        lines.append(f"{_ANSI_HDR}{border}{header_inner}"
                     + "═" * max(0, 71 - len(border) - len(header_inner))
                     + _ANSI_RESET)

    name_width = max([len(n) for n in results.keys()] + [16])
    name_width = max(name_width, 20)

    for name, r in results.items():
        lift = r["lift"]
        std = r["std"]
        sigma = r["sigma"]
        bb100 = icm_lift_to_bb_per_100(lift, structure)
        if colored:
            color = _ANSI_POS if lift >= 0 else _ANSI_NEG
            arrow = "▲" if lift >= 0 else "▼"
            bb_part = f"  {bb100:+7.1f} bb/100" if bb100 is not None else ""
            lines.append(
                f"  vs {name:<{name_width}}  "
                f"{color}{lift:+8.4f} ICM{_ANSI_RESET}{bb_part}  "
                f"sigma {sigma:5.2f}   {color}{arrow}{_ANSI_RESET}"
            )
        else:
            bb_part = f" ({bb100:+.1f} bb/100)" if bb100 is not None else ""
            lines.append(
                f"[iter {iter_num:>4}] lift_vs_{name}: "
                f"{lift:+.4f} ICM{bb_part}  "
                f"{n_hands}h  std={std:.4f} sigma={sigma:.2f}"
            )

    if placeholder_self_label is not None:
        if colored:
            lines.append(
                f"  vs {placeholder_self_label:<{name_width}}  "
                f"{_ANSI_DIM}(no prior checkpoint){_ANSI_RESET}"
            )
        else:
            lines.append(
                f"[iter {iter_num:>4}] lift_vs_{placeholder_self_label}: "
                f"(no prior checkpoint)"
            )

    if colored:
        lines.append(f"{_ANSI_HDR}{'═' * 71}{_ANSI_RESET}")

    return "\n".join(lines)


def icm_lift_to_bb_per_100(
    icm_lift_per_hand: float,
    structure: Any,
) -> Optional[float]:
    """Convert ICM-equity-delta per hand to bb/100 (rule-of-thumb).

    The CFR solver optimizes ICM-equity-delta (in normalized buy-in units).
    bb/100 is a chip-EV reading used as a sanity-check secondary metric.
    Formula:
        bb/100 = icm_lift × (payout_per_buyin / buy_in_dollars)
                          × (starting_chips / big_blind)
                          × 100

    The conversion assumes the structure exposes payouts_dollars (in $),
    buy_in_dollars, starting_chips, and at least one BlindLevel in
    blind_schedule (uses big_blind from blind_schedule[0] as the reference).
    Returns None when the structure cannot provide a meaningful big_blind
    (e.g., missing blind_schedule), so callers can omit the parenthetical
    cleanly rather than print a nonsense number.
    """
    try:
        payout_per_buyin = float(structure.payouts_dollars[0])
        buy_in = float(structure.buy_in_dollars)
        starting_chips = float(structure.starting_chips)
        big_blind = float(structure.blind_schedule[0].big_blind)
    except (AttributeError, IndexError, TypeError):
        return None
    if buy_in <= 0 or big_blind <= 0:
        return None
    return (
        icm_lift_per_hand
        * (payout_per_buyin / buy_in)
        * (starting_chips / big_blind)
        * 100.0
    )


def _parse_anchor_spec(spec: str) -> tuple[str, str]:
    """'name=path' -> (name, path); bare 'path' -> (derived_name, path)."""
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name.strip(), path.strip()
    # Derive a readable name: <run-tag-ish>/<ckpt> or basename.
    base = os.path.basename(spec)
    return base, spec


def _load_anchor_policy(solver: Any, spec: str, is_shanky: bool):
    """Load (and cache on the solver) an anchor as a Policy."""
    name, path = _parse_anchor_spec(spec)
    cache = solver._mini_eval_anchor_cache
    if path in cache:
        return cache[path]
    if is_shanky:
        from src.nlhe.scripted_bots.policy import ShankyProfilePolicy
        policy = ShankyProfilePolicy(name=name, profile_path=path)
    else:
        from scripts.eval_pool import CheckpointPolicy
        policy = CheckpointPolicy(
            name, path, solver.abstraction, solver.tournament_structure
        )
    cache[path] = policy
    return policy


def run_mini_eval(
    solver: Any,
    iter_num: int,
    self_anchor_path: Optional[str] = None,
    self_anchor_label: Optional[str] = None,
) -> dict:
    """Snapshot the live solver and head-to-head it against the anchors.

    Returns {anchor_name: {"lift": float, "std": float, "sigma": float}}.
    `lift` is the challenger-minus-opponent ICM-equity-delta (evaluate_matchup's
    `diff`); positive = challenger ahead.

    Self-anchor (Phase 3 dashboard extension): when self_anchor_path is set
    AND the file exists, the challenger is also evaluated against a
    CheckpointPolicy loaded from that path (a frozen earlier snapshot of the
    solver — typically the checkpoint from `mini_eval_every` iters ago,
    surfaced as `lift_vs_self_iter_XXXX`). self_anchor_label becomes the
    result-dict key (e.g. "self_iter_0200"). Skipped silently when the
    file is missing — the caller (solver._maybe_run_mini_eval) emits a
    placeholder line for log continuity.
    """
    from scripts.eval_pool import CheckpointPolicy, evaluate_matchup

    cfg = solver.cfg
    structure = solver.tournament_structure
    if structure is None:
        raise ValueError(
            "mini_eval requires tournament_structure_path to be set (anchors "
            "and challenger are evaluated under the tournament structure)"
        )

    # Snapshot the live nets (slim — weights only) and wrap as a Policy.
    solver.save_checkpoint(CHALLENGER_TMP, slim=True)
    challenger = CheckpointPolicy(
        "challenger", CHALLENGER_TMP, solver.abstraction, structure
    )

    # Fixed eval seed derived from the training seed — isolated from self.rng.
    seed = cfg.seed + 200 + iter_num
    n_hands = cfg.mini_eval_n_hands
    results: dict = {}

    for spec in (cfg.mini_eval_anchors or []):
        name, _ = _parse_anchor_spec(spec)
        opp = _load_anchor_policy(solver, spec, is_shanky=False)
        r = evaluate_matchup(challenger, opp, structure, n_hands, seed=seed,
                             log_every=n_hands + 1)
        results[name] = {"lift": r["diff"], "std": r["stderr"], "sigma": r["sigma"]}

    # One rotating Shanky profile per snapshot (cycles through the list).
    rotation = cfg.mini_eval_shanky_rotation or []
    if rotation:
        idx = (iter_num // max(cfg.mini_eval_every, 1)) % len(rotation)
        spec = rotation[idx]
        sname, _ = _parse_anchor_spec(spec)
        name = f"shanky-{sname}"
        opp = _load_anchor_policy(solver, spec, is_shanky=True)
        r = evaluate_matchup(challenger, opp, structure, n_hands, seed=seed,
                             log_every=n_hands + 1)
        results[name] = {"lift": r["diff"], "std": r["stderr"], "sigma": r["sigma"]}

    # Self-anchor: lift vs frozen self from a prior checkpoint.
    if (
        self_anchor_path is not None
        and self_anchor_label is not None
        and os.path.exists(self_anchor_path)
    ):
        self_opp = CheckpointPolicy(
            self_anchor_label, self_anchor_path,
            solver.abstraction, structure,
        )
        r = evaluate_matchup(challenger, self_opp, structure, n_hands, seed=seed,
                             log_every=n_hands + 1)
        results[self_anchor_label] = {
            "lift": r["diff"], "std": r["stderr"], "sigma": r["sigma"],
        }

    return results
