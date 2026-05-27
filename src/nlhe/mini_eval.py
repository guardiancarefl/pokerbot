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
from typing import Any

# Fixed path: assumes a single training run per host. Concurrent training
# runs on the same host would clobber each other's challenger snapshot.
# Add a PID/run-dir suffix if multi-run support becomes needed.
CHALLENGER_TMP = "/tmp/step7_mini_eval_challenger.pt"


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


def run_mini_eval(solver: Any, iter_num: int) -> dict:
    """Snapshot the live solver and head-to-head it against the anchors.

    Returns {anchor_name: {"lift": float, "std": float, "sigma": float}}.
    `lift` is the challenger-minus-opponent ICM-equity-delta (evaluate_matchup's
    `diff`); positive = challenger ahead.
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

    return results
