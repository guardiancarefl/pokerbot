"""Config dataclass and YAML loader for Leduc Deep CFR training.

All training hyperparameters live in a YAML file under configs/, are loaded
into a TrainConfig dataclass here, and are passed as a single object through
the rest of the pipeline. CLI args can override individual fields.

The dataclass is the source of truth for what's configurable. Adding a new
hyperparameter means adding a field here, then referencing it in solver.py.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    """All hyperparameters for a Deep CFR training run.

    Defaults match a small but real Leduc training run. Override via YAML
    config file or CLI flags.
    """

    # --- Algorithm ---
    iterations: int = 100
    num_traversals: int = 40
    learning_rate: float = 1e-3
    memory_capacity: int = 1_000_000

    # --- Networks ---
    advantage_network_layers: list[int] = field(default_factory=lambda: [64, 64])
    policy_network_layers: list[int] = field(default_factory=lambda: [64, 64])
    advantage_train_steps: int = 500
    policy_train_steps: int = 1000
    batch_size_advantage: int = 128
    batch_size_strategy: int = 1024

    # --- Reproducibility ---
    seed: int = 42

    # --- Evaluation ---
    skip_exploitability: bool = False

    # --- Run metadata ---
    run_name: str | None = None  # appended to timestamped run dir if set

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainConfig":
        """Load config from a YAML file. Unknown keys raise ValueError."""
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainConfig":
        """Build config from a dict, validating that keys match the dataclass."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown config keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid_fields)}"
            )
        return cls(**data)

    def merge_overrides(self, overrides: dict[str, Any]) -> "TrainConfig":
        """Return a new TrainConfig with CLI overrides applied.

        Only non-None overrides are applied, so CLI args without explicit
        values (None) don't clobber YAML values.
        """
        base = asdict(self)
        for k, v in overrides.items():
            if v is not None:
                if k not in base:
                    raise ValueError(f"Unknown override key: {k}")
                base[k] = v
        return TrainConfig.from_dict(base)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
