"""Dashboard / mini-eval regression: self-anchor + tail-friendly log format.

Tiny end-to-end test (6 iters @ mini_eval_every=2 → 3 eval cycles):
  - lift.log exists in the run dir parent of checkpoints.
  - Each line matches the expected single-line tail-friendly format.
  - Iter 2 emits the "(no prior checkpoint)" self-anchor placeholder.
  - Iter 4 and iter 6 emit real lift_vs_self_iter_XXXX numbers
    (challenger vs the previous cycle's checkpoint).
  - metrics["mini_eval"] has 3 records covering iters 2/4/6; the
    iter-4 and iter-6 records' results dicts include the self_iter_XXXX
    key.

Sample size is tiny (mini_eval_n_hands=20) — this gates wiring, NOT
statistical validity.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pyspiel
import pytest
import yaml

from src.nlhe.abstraction import Abstraction
from src.nlhe.game_strings import PokerGameConfig
from src.nlhe.solver6 import DeepCFR6MaxSolver, TrainConfig6Max


REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_CONFIG_PATH = REPO_ROOT / "configs" / "baseline_seq.yaml"
TOURNAMENT_STRUCTURE_PATH = (
    REPO_ROOT / "configs" / "ignition_double_up_6max_turbo.yaml"
)
SHANKY_FIXTURE = REPO_ROOT / "tests" / "scripted_bots_fixtures" / "littlegreen2.txt"
ANCHOR_CKPT = (
    REPO_ROOT / "runs" / "baseline_fork_A" / "checkpoints" / "ckpt_iter_0016.pt"
)


# Tail-friendly single-line formats.
_REAL_LINE_RE = re.compile(
    r"^\[iter\s+\d+\] lift_vs_[\w.\-]+: "
    r"[+-]\d+\.\d+ ICM"
    r"( \([+-]\d+\.\d+ bb/100\))?"
    r"\s+\d+h\s+std=\d+\.\d+\s+sigma=\d+\.\d+$"
)
_PLACEHOLDER_LINE_RE = re.compile(
    r"^\[iter\s+\d+\] lift_vs_self_iter_\d+: \(no prior checkpoint\)$"
)


@pytest.fixture(scope="module")
def dashboard_run(tmp_path_factory):
    if not BASELINE_CONFIG_PATH.exists():
        pytest.skip(f"baseline_seq.yaml missing")
    if not TOURNAMENT_STRUCTURE_PATH.exists():
        pytest.skip(f"tournament structure missing")
    if not SHANKY_FIXTURE.exists():
        pytest.skip(f"Shanky fixture missing")
    if not ANCHOR_CKPT.exists():
        pytest.skip(
            f"anchor checkpoint missing at {ANCHOR_CKPT}; "
            f"regenerate runs/baseline_fork_A via configs/baseline_seq.yaml"
        )

    out = tmp_path_factory.mktemp("dashboard_self_anchor") / "run"
    out.mkdir(parents=True)
    ckpt_dir = out / "checkpoints"

    base = yaml.safe_load(open(BASELINE_CONFIG_PATH))
    tc_kwargs = {
        k: v for k, v in base.items()
        if k not in ("tag", "abstraction_path", "checkpoint_every")
    }
    tc_kwargs.update({
        "tournament_structure_path": str(TOURNAMENT_STRUCTURE_PATH),
        "hidden_dim": [32, 32],
        "n_iterations": 6,
        "traversals_per_iter": 10,
        "train_steps_per_iter": 5,
        "batch_size": 8,
        "mini_eval_enabled": True,
        "mini_eval_every": 2,
        "mini_eval_n_hands": 20,
        "mini_eval_anchors": [f"baseA={ANCHOR_CKPT}"],
        "mini_eval_shanky_rotation": [f"littlegreen2={SHANKY_FIXTURE}"],
    })
    tc = TrainConfig6Max(**tc_kwargs)

    game_str = PokerGameConfig(
        num_players=6,
        starting_stack=tc.starting_stack,
        big_blind=tc.big_blind,
        small_blind=tc.small_blind,
    ).to_universal_poker_string()
    game = pyspiel.load_game(game_str)
    abst = Abstraction.load(base["abstraction_path"])
    solver = DeepCFR6MaxSolver(game=game, abstraction=abst, config=tc)
    metrics = solver.train(checkpoint_dir=ckpt_dir, checkpoint_every=2)

    return {
        "run_dir": out,
        "ckpt_dir": ckpt_dir,
        "lift_log": out / "lift.log",
        "metrics": metrics,
    }


def test_lift_log_exists(dashboard_run):
    assert dashboard_run["lift_log"].exists(), "lift.log not created"


def test_lift_log_lines_match_expected_format(dashboard_run):
    """Every non-empty line in lift.log matches either the real-numbers
    single-line format or the (no prior checkpoint) placeholder format."""
    text = dashboard_run["lift_log"].read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert lines, "lift.log empty"
    bad = []
    for ln in lines:
        if not (_REAL_LINE_RE.match(ln) or _PLACEHOLDER_LINE_RE.match(ln)):
            bad.append(ln)
    assert not bad, f"lines do not match expected format: {bad}"


def test_first_cycle_emits_self_anchor_placeholder(dashboard_run):
    text = dashboard_run["lift_log"].read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    iter2_self = [ln for ln in lines if ln.startswith("[iter    2]") and "lift_vs_self" in ln]
    assert iter2_self, "iter 2 self-anchor line missing"
    assert "no prior checkpoint" in iter2_self[0], (
        f"iter 2 self-anchor should be placeholder, got: {iter2_self[0]}"
    )


def test_subsequent_cycles_emit_real_self_anchor_numbers(dashboard_run):
    text = dashboard_run["lift_log"].read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for iter_label, prev_label in [("[iter    4]", "self_iter_0002"),
                                    ("[iter    6]", "self_iter_0004")]:
        selflines = [ln for ln in lines if ln.startswith(iter_label) and prev_label in ln]
        assert selflines, f"{iter_label} self-anchor for {prev_label} missing"
        assert _REAL_LINE_RE.match(selflines[0]), (
            f"{iter_label} self-anchor line format wrong: {selflines[0]}"
        )


def test_metrics_mini_eval_records_correct_iters(dashboard_run):
    records = dashboard_run["metrics"]["mini_eval"]
    iters = [r["iter"] for r in records]
    assert iters == [2, 4, 6], f"expected iters [2,4,6], got {iters}"


def test_metrics_mini_eval_records_include_self_anchor_after_first_cycle(dashboard_run):
    records = dashboard_run["metrics"]["mini_eval"]
    # iter 2 (first cycle): no self-anchor in results dict
    assert "self_iter_0000" not in records[0]["results"]
    # iter 4 and 6: real self-anchor in results
    assert "self_iter_0002" in records[1]["results"], (
        f"iter 4 results should contain self_iter_0002, got {list(records[1]['results'].keys())}"
    )
    assert "self_iter_0004" in records[2]["results"], (
        f"iter 6 results should contain self_iter_0004, got {list(records[2]['results'].keys())}"
    )
