"""Checkpoint save/load for Deep CFR policy networks.

A checkpoint captures everything needed to:
  - reload the trained policy network for inference or evaluation later
  - know exactly what config produced it
  - know what the training metrics were

We deliberately don't try to capture full solver state (advantage buffers,
memory, etc.) for resumability — OpenSpiel's DeepCFRSolver doesn't expose
the hooks for that, and Phase 1 doesn't need it. If we want resumable
training later we'll fork the solver, not add it here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: Path,
    policy_network: torch.nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Save policy network + config + metrics to a single .pt file.

    Companion config.json and metrics.json are written next to the .pt file
    for human inspection without needing torch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "policy_network_state_dict": policy_network.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )

    # Also write companion JSON files for human inspection.
    (path.parent / "config.json").write_text(json.dumps(config, indent=2))
    (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoint dict. Caller is responsible for reconstructing the
    network architecture and calling load_state_dict()."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)
