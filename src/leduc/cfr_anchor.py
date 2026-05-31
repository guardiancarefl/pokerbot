"""Tabular CFR+ Nash anchor for Leduc (Family-2 adaptive-policy proof).

Why CFR+ and not Deep CFR: Leduc is tiny (~1k information states), so tabular
CFR+ drives exploitability to near-zero in seconds — a *pristine* near-Nash
anchor. The Phase-1 Deep CFR checkpoints plateaued at ~430 mbb/g (undertrained,
see runs/leduc_20260521_210552_phase1_take2). This module gives the clean GTO
bench the adaptive policy is (a) distilled toward when it has no read and
(b) measured against for the safety half of the pass bar.

Uses OpenSpiel's CFRPlusSolver; the *average* policy is the anchor. Validated
with src.leduc.evaluate.exploitability_mbb (exact full-tree exploitability).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr

from src.leduc.evaluate import exploitability_mbb


@dataclass
class AnchorResult:
    iterations: int
    exploitability_mbb: float
    history: list            # [(iteration, exploitability_mbb), ...]
    avg_policy: object       # OpenSpiel TabularPolicy (the average strategy)
    game: object


def train_cfr_plus_anchor(iterations: int = 1000, eval_every: int = 100,
                          game_name: str = "leduc_poker") -> AnchorResult:
    """Run CFR+ for `iterations` full-tree passes; return the average policy +
    its exact exploitability (mbb/game). Exploitability is logged every
    `eval_every` iterations so the descent is visible."""
    game = pyspiel.load_game(game_name)
    solver = cfr.CFRPlusSolver(game)
    history = []
    for it in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if it % eval_every == 0 or it == iterations:
            expl = exploitability_mbb(game, solver.average_policy().action_probabilities)
            history.append((it, expl))
    avg = solver.average_policy()
    final = exploitability_mbb(game, avg.action_probabilities)
    return AnchorResult(iterations, final, history, avg, game)


def export_anchor_table(avg_policy) -> dict:
    """Portable {info_state_string: {action_id: prob}} of the average strategy.

    Keyed by OpenSpiel `information_state_string` so Step-3 distillation can look
    up the GTO target for any state via `state.information_state_string()`.
    Only legal actions are emitted (probs renormalize trivially since the array
    is already a distribution over the legal mask)."""
    arr = np.asarray(avg_policy.action_probability_array)
    mask = np.asarray(avg_policy.legal_actions_mask)
    table = {}
    for info, idx in avg_policy.state_lookup.items():
        probs = arr[idx]
        legal = [a for a in range(probs.shape[0]) if mask[idx][a]]
        table[info] = {int(a): float(probs[a]) for a in legal}
    return table


def save_anchor(result: AnchorResult, out_dir) -> dict:
    """Write anchor_table.json (distillation target), avg_policy_arrays.pkl
    (reconstructable TabularPolicy state), and metrics.json. Returns metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = export_anchor_table(result.avg_policy)
    (out_dir / "anchor_table.json").write_text(json.dumps(table))
    with open(out_dir / "avg_policy_arrays.pkl", "wb") as f:
        pickle.dump({
            "state_lookup": dict(result.avg_policy.state_lookup),
            "action_probability_array": np.asarray(result.avg_policy.action_probability_array),
            "legal_actions_mask": np.asarray(result.avg_policy.legal_actions_mask),
        }, f)
    metrics = {
        "iterations": result.iterations,
        "exploitability_mbb_per_game": result.exploitability_mbb,
        "exploitability_history": [{"iteration": i, "exploitability_mbb": e}
                                   for i, e in result.history],
        "n_info_states": len(table),
        "anchor_method": "tabular_cfr_plus",
        "nash_bar_mbb": 5.0,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
